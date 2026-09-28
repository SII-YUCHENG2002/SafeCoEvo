"""Execute Agent-SafetyBench cases in their native environments."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from agent_safetybench_harness import SafeEnvironmentSession
from benchmark_runtime import ROOT, UnifiedCase, native_tool_call, tool_maps, tool_protocol_error
from target_retry import (
    DEFAULT_FALLBACK_API_KEY,
    DEFAULT_FALLBACK_BASE_URL,
    DEFAULT_FALLBACK_MODEL,
    completion_with_retry_and_fallback,
)
def run_case(
    case: UnifiedCase,
    *,
    model: str,
    base_url: str,
    api_key_env: str = "INF_API_KEY",
    max_tokens: int = 4096,
    timeout_seconds: float = 180.0,
    target_request_retries: int = 3,
    target_retry_sleep_seconds: float = 5.0,
    target_fallback_model: str = DEFAULT_FALLBACK_MODEL,
    target_fallback_base_url: str = DEFAULT_FALLBACK_BASE_URL,
    target_fallback_api_key: str = DEFAULT_FALLBACK_API_KEY,
    target_fallback_request_retries: int = 2,
    harness_session: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if case.source != "agent_safetybench":
        raise ValueError(f"Wrong executor for source {case.source}")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing {api_key_env}")
    released = case.source_record["source_payload"]["released_record"]
    dataset_root = Path(os.environ.get(
        "SAFECOEVO_AGENT_SAFETYBENCH_ROOT",
        str(ROOT / "external/Agent-SafetyBench"),
    ))
    session = SafeEnvironmentSession(dataset_root, released["environments"])
    public_to_native, _ = tool_maps(case)
    messages: list[dict[str, Any]] = harness_session.agent_input()["messages"]
    trace: list[dict[str, Any]] = []
    errors = 0
    final = ""
    status = "exceeded_turn_limit"
    started = time.perf_counter()
    for turn in range(int(case.agent["runtime"]["max_turns"])):
        request_context = harness_session.prepare_model_request()
        messages = request_context["messages"]
        request_tools = request_context["tools"]
        request = {
            "model": model,
            "messages": messages,
            "tools": request_tools,
            "parallel_tool_calls": False,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "seed": 0,
            "extra_body": {"chat_template_kwargs": {"thinking": False}},
        }
        reply = completion_with_retry_and_fallback(
            request=request,
            primary_model=model,
            primary_base_url=base_url,
            primary_api_key=api_key,
            timeout_seconds=timeout_seconds,
            primary_retries=target_request_retries,
            fallback_model=target_fallback_model,
            fallback_base_url=target_fallback_base_url,
            fallback_api_key=target_fallback_api_key,
            fallback_retries=target_fallback_request_retries,
            retry_sleep_seconds=target_retry_sleep_seconds,
        )
        calls = list(reply.tool_calls or [])
        if not calls:
            final = reply.content or ""
            from safecoevo_runtime import CandidateAction

            execution = harness_session.submit_action(
                CandidateAction(action_id=f"turn-{turn + 1}-final", kind="final", text=final)
            )
            if execution.status == "withheld":
                messages = harness_session.agent_input()["messages"]
                trace.append(
                    {
                        "turn": turn + 1,
                        "kind": "final_withheld",
                        "content": final,
                        "execution_status": execution.status,
                        "intercepted_by": list(execution.intercepted_by),
                    }
                )
                continue
            if execution.effective_action is not None:
                final = execution.effective_action.text
            messages = harness_session.agent_input()["messages"]
            trace.append({"turn": turn + 1, "kind": "final", "content": final})
            status = "completed"
            break
        call = calls[0]
        raw_args = call.function.arguments or "{}"
        try:
            proposed_args = json.loads(raw_args)
            if not isinstance(proposed_args, dict):
                raise ValueError("Tool arguments must be a JSON object")
        except (json.JSONDecodeError, TypeError, ValueError):
            proposed_args = {"raw_arguments": raw_args}
        from safecoevo_runtime import CandidateAction, ToolCall

        execution = harness_session.submit_action(
            CandidateAction(
                action_id=f"turn-{turn + 1}-tool-{call.id or 'unknown'}",
                kind="tool",
                text=reply.content or "",
                tool_call=ToolCall(
                    call_id=str(call.id or f"call-{turn + 1}"),
                    name=str(call.function.name),
                    arguments=proposed_args,
                ),
            )
        )
        if execution.status == "internal":
            result = execution.synthetic_tool_result or "[Harness-internal tool completed]"
            harness_session.record_tool_result(result)
            messages = harness_session.agent_input()["messages"]
            trace.append(
                {
                    "turn": turn + 1,
                    "kind": "harness_internal_tool",
                    "public_tool": call.function.name,
                    "arguments": proposed_args,
                    "result": result,
                    "execution_status": execution.status,
                    "intercepted_by": list(execution.intercepted_by),
                }
            )
            continue
        if execution.status == "withheld":
            result = execution.synthetic_tool_result or "[Tool call withheld by SafeCoEvo processor]"
            harness_session.record_tool_result(result)
            messages = harness_session.agent_input()["messages"]
            trace.append(
                {
                    "turn": turn + 1,
                    "kind": "tool_withheld",
                    "public_tool": call.function.name,
                    "arguments": proposed_args,
                    "result": result,
                    "status": "withheld",
                    "execution_status": execution.status,
                    "intercepted_by": list(execution.intercepted_by),
                }
            )
            continue
        effective_tool = execution.effective_action.tool_call if execution.effective_action else None
        if effective_tool is None:
            raise RuntimeError("Harness returned an executable tool disposition without a tool call")
        execution_name = effective_tool.name
        execution_args = effective_tool.arguments
        try:
            args = execution_args
            native, native_args = native_tool_call(execution_name, args, public_to_native)
            result = session.call(native, native_args)
            tool_status = "ok"
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            args, result, tool_status = {}, {"success": False, "message": tool_protocol_error(exc)}, "protocol_error"
            errors += 1
        text = json.dumps(result, ensure_ascii=False, default=str)
        harness_session.record_tool_result(text)
        messages = harness_session.agent_input()["messages"]
        trace.append({"turn": turn + 1, "kind": "tool", "public_tool": call.function.name, "effective_public_tool": execution_name, "arguments": args, "result": text, "status": tool_status})
    return ({"task_id": case.task_id, "source": case.source, "status": status, "fulfillable": int(released["fulfillable"]), "safety_label_available": False, "turn_count": len(trace), "tool_call_count": sum(step["kind"] == "tool" for step in trace), "protocol_error_count": errors, "elapsed_seconds": time.perf_counter() - started}, {"messages": messages, "trace": trace, "final_text": final, "environment": released["environments"]})
