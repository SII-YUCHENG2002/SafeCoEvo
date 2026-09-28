#!/usr/bin/env python3
"""Serve the local AgentDoG model through an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Keep TorchDynamo/Inductor enabled for Qwen3.5 throughput. Request
# serialization, token caps, and allocator cleanup bound request-local memory.

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
if TYPE_CHECKING:
    from local_agentdog_diagnoser import LocalAgentDoGDiagnoser

DEFAULT_MODEL_PATH = Path(os.environ.get(
    "AGENTDOG_MODEL_PATH",
    str(ROOT / "models/AgentDoG1.5-Unified-Qwen3.5-4B"),
))


class AgentDoGServer:
    def __init__(
        self,
        diagnoser: LocalAgentDoGDiagnoser,
        model_name: str,
        *,
        max_new_tokens: int,
        max_concurrent_generations: int = 2,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if max_concurrent_generations < 1:
            raise ValueError("max_concurrent_generations must be positive")
        self.diagnoser = diagnoser
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_concurrent_generations = max_concurrent_generations
        self._generation_slots = threading.BoundedSemaphore(max_concurrent_generations)
        self._model_generate_lock = threading.Lock()
        self._request_count = 0

    @staticmethod
    def _cuda_memory() -> list[dict[str, Any]]:
        import torch

        if not torch.cuda.is_available():
            return []
        return [
            {
                "device": index,
                "allocated_mib": round(torch.cuda.memory_allocated(index) / 2**20, 1),
                "reserved_mib": round(torch.cuda.memory_reserved(index) / 2**20, 1),
                "max_allocated_mib": round(torch.cuda.max_memory_allocated(index) / 2**20, 1),
            }
            for index in range(torch.cuda.device_count())
        ]

    @staticmethod
    def _release_request_memory() -> None:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                with torch.cuda.device(index):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(index)
        # Release host-side allocator arenas when glibc exposes malloc_trim.
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (AttributeError, OSError):
            pass

    def generate(self, messages: list[dict[str, Any]], max_tokens: int, temperature: float) -> dict[str, Any]:
        # Support standard OpenAI chat clients: generate from the
        # supplied messages verbatim instead of rebuilding the project prompt.
        tokenizer = self.diagnoser.tokenizer
        model = self.diagnoser.model
        # AgentDoG's Qwen template otherwise emits a long <think> section before
        # its verdict, which can consume the bounded response budget and hide JSON.
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        requested_max_tokens = max(1, int(max_tokens))
        effective_max_tokens = min(requested_max_tokens, self.max_new_tokens)
        # Transformers + FLA/Triton Autotuner mutate shared kernel state during
        # generation. Concurrent calls on one model instance corrupt that state.
        with self._generation_slots, self._model_generate_lock:
            import torch

            self._request_count += 1
            request_number = self._request_count
            before_memory = self._cuda_memory()
            inputs = tokenizer([text], return_tensors="pt")
            first_device = next(model.parameters()).device
            inputs = {key: value.to(first_device) for key, value in inputs.items()}
            start = time.perf_counter()
            try:
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=effective_max_tokens,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else None,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        # Restore the model's fast generation cache. Request
                        # serialization, the hard token cap, and post-request
                        # allocator cleanup bound its lifetime and memory use.
                        use_cache=True,
                    )
                latency = time.perf_counter() - start
                prompt_tokens = int(inputs["input_ids"].shape[1])
                output_ids = output[0][prompt_tokens:].tolist()
                content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            finally:
                # Delete request-owned CUDA tensors before empty_cache so the
                # allocator can return unused blocks to the driver.
                if "output" in locals():
                    del output
                if "inputs" in locals():
                    del inputs
                self._release_request_memory()
            after_memory = self._cuda_memory()
            print(
                json.dumps(
                    {
                        "event": "agentdog_request_memory",
                        "request": request_number,
                        "requested_max_tokens": requested_max_tokens,
                        "effective_max_tokens": effective_max_tokens,
                        "before": before_memory,
                        "after_cleanup": after_memory,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return {
            "id": f"chatcmpl-agentdog-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(output_ids),
                "total_tokens": prompt_tokens + len(output_ids),
            },
            "agentdog_server": {
                "latency_seconds": latency,
                "request_number": request_number,
                "requested_max_tokens": requested_max_tokens,
                "effective_max_tokens": effective_max_tokens,
                "cuda_memory_after_cleanup": after_memory,
                "generation_serialized": True,
                "max_concurrent_generations": self.max_concurrent_generations,
                "use_cache": True,
            },
        }


def make_handler(server_state: AgentDoGServer):
    class Handler(BaseHTTPRequestHandler):
        server_version = "LocalAgentDoGOpenAI/0.1"

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") == "/v1/models":
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": server_state.model_name, "object": "model", "owned_by": "local"}],
                    },
                )
                return
            self._send_json(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send_json(404, {"error": {"message": "not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                messages = payload.get("messages") or []
                if not isinstance(messages, list):
                    raise ValueError("messages must be a list")
                max_tokens = int(payload.get("max_tokens") or payload.get("max_completion_tokens") or 512)
                temperature = float(payload.get("temperature") or 0)
                response = server_state.generate(messages, max_tokens=max_tokens, temperature=temperature)
                self._send_json(200, response)
            except Exception as exc:  # intentionally broad: return OpenAI-style error
                traceback.print_exc(file=sys.stderr)
                self._send_json(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-api-config", type=Path, default=ROOT / "configs/runtime_api_config.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-memory-gib", type=int, default=4)
    parser.add_argument("--max-concurrent-generations", type=int, default=2)
    args = parser.parse_args()

    from runtime_api_config import require_runtime_api_config

    try:
        require_runtime_api_config(args.runtime_api_config)
    except ValueError as exc:
        parser.error(str(exc))

    from local_agentdog_diagnoser import LocalAgentDoGDiagnoser

    print(f"Loading AgentDoG from {args.model_path}", flush=True)
    diagnoser = LocalAgentDoGDiagnoser(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        max_memory_gib=args.max_memory_gib,
    )
    print("Model loaded; hf_device_map=", diagnoser.hf_device_map, flush=True)
    httpd = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(
            AgentDoGServer(
                diagnoser,
                os.environ["AGENTDOG_MODEL"],
                max_new_tokens=args.max_new_tokens,
                max_concurrent_generations=args.max_concurrent_generations,
            )
        ),
    )
    print(f"Serving OpenAI-compatible AgentDoG at http://{args.host}:{args.port}/v1", flush=True)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
