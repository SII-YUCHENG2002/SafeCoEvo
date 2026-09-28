#!/usr/bin/env python3
"""Validate prerequisites and launch the local AgentDoG API server."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LOG = ROOT / "logs/agentdog_server_current.out"
PID = ROOT / "logs/agentdog_server_current.pid"
DEFAULT_SERVER_PYTHON = Path(sys.executable)
DEFAULT_MODEL_PATH = Path(os.environ.get(
    "AGENTDOG_MODEL_PATH",
    str(ROOT / "models/AgentDoG1.5-Unified-Qwen3.5-4B"),
))


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(cmd, 127, "", str(exc))


def existing_server_pid() -> int | None:
    proc = run(["ps", "-eo", "pid=,cmd="])
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if "serve_agentdog.py" not in line:
            continue
        if "start_agentdog_server.py" in line:
            continue
        parts = line.strip().split(maxsplit=1)
        if parts and parts[0].isdigit():
            return int(parts[0])
    return None


def gpu_status() -> list[dict[str, int]]:
    proc = run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"])
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "nvidia-smi failed")
    rows = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        idx, used, total = [item.strip() for item in line.split(",")]
        rows.append({"index": int(idx), "memory_used_mb": int(used), "memory_total_mb": int(total)})
    return rows


def python_runtime(python: Path, cuda_visible_devices: str | None) -> dict[str, Any]:
    """Check the interpreter that will load the model, without allocating it."""

    if not python.is_file():
        return {"ok": False, "error": f"interpreter does not exist: {python}"}
    probe = (
        "import json, sys, torch, transformers; "
        "print(json.dumps({'python': sys.executable, 'torch': torch.__version__, "
        "'transformers': transformers.__version__, 'cuda_available': torch.cuda.is_available(), "
        "'cuda_device_count': torch.cuda.device_count()}))"
    )
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    proc = subprocess.run([str(python), "-c", probe], cwd=ROOT, text=True, capture_output=True, check=False, env=env)
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.strip() or proc.stdout.strip() or "runtime probe failed"}
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": f"invalid runtime probe output: {proc.stdout.strip()}"}
    result["ok"] = bool(result.get("cuda_available"))
    if not result["ok"]:
        result["error"] = "selected Python has no CUDA runtime"
    return result


def select_gpus(available: list[dict[str, int]], requested: str | None) -> list[dict[str, int]]:
    if requested is None:
        return available
    try:
        wanted = [int(value.strip()) for value in requested.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("--cuda-visible-devices must be a comma-separated list of GPU indices") from exc
    if not wanted or len(set(wanted)) != len(wanted):
        raise ValueError("--cuda-visible-devices must contain one or more unique GPU indices")
    by_index = {row["index"]: row for row in available}
    missing = sorted(set(wanted) - set(by_index))
    if missing:
        raise ValueError(f"requested GPU indices are unavailable: {missing}")
    return [by_index[index] for index in wanted]


def checks(python: Path, model_path: Path, cuda_visible_devices: str | None) -> dict[str, Any]:
    detected_gpus = gpu_status()
    gpus = select_gpus(detected_gpus, cuda_visible_devices)
    return {
        "server_python": str(python),
        "server_runtime": python_runtime(python, ",".join(str(row["index"]) for row in gpus)),
        "model_path": str(model_path),
        "model_path_exists": model_path.is_dir(),
        "model_config_exists": (model_path / "config.json").is_file(),
        "detected_gpu_count": len(detected_gpus),
        "gpu_count": len(gpus),
        "selected_gpu_indices": [row["index"] for row in gpus],
        "gpus": gpus,
        "existing_server_pid": existing_server_pid(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-api-config", type=Path, default=ROOT / "configs/runtime_api_config.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-memory-gib", type=int, default=8)
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional comma-separated physical GPU indices used only by the AgentDoG server.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=DEFAULT_SERVER_PYTHON,
        help="CUDA-capable Python used by the isolated AgentDoG server process.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    from runtime_api_config import require_runtime_api_config

    try:
        require_runtime_api_config(args.runtime_api_config)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        status = checks(args.python, args.model_path, args.cuda_visible_devices)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"event": "agentdog_server_preflight", **status}, ensure_ascii=False, indent=2, sort_keys=True))
    blockers = []
    if not status["server_runtime"].get("ok"):
        blockers.append(f"selected server runtime is unusable: {status['server_runtime'].get('error', 'CUDA unavailable')}")
    if not status["model_path_exists"] or not status["model_config_exists"]:
        blockers.append("AgentDoG model path or config.json is missing")
    if status["existing_server_pid"]:
        blockers.append(f"AgentDoG server already running: pid={status['existing_server_pid']}")
    if blockers:
        print(json.dumps({"event": "agentdog_server_not_started", "blockers": blockers}, ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    if args.dry_run:
        print(json.dumps({"event": "agentdog_server_dry_run_ok", "log": str(LOG)}, ensure_ascii=False, sort_keys=True))
        return 0

    LOG.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.python),
        "scripts/serve_agentdog.py",
        "--runtime-api-config", str(args.runtime_api_config.resolve()),
        "--host", args.host,
        "--port", str(args.port),
        "--max-new-tokens", str(args.max_new_tokens),
        "--max-memory-gib", str(args.max_memory_gib),
        "--model-path", str(args.model_path),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(row["index"]) for row in status["gpus"])
    with LOG.open("ab") as log_f:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True)
    PID.write_text(str(proc.pid) + "\n", encoding="utf-8")
    print(json.dumps({"event": "agentdog_server_started", "pid": proc.pid, "log": str(LOG)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
