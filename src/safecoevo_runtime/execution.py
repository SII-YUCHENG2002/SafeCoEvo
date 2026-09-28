"""Source-preserving execution adapters for the SafeCoEvo runtime."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import CandidateAction, GuardVerdict, TaskOutcome
from .dataset import SafeCoEvoCase
from .events import PublicTask
from .runtime import HarnessSession
from .trace import TraceStore


def run_case_directory_name(*, ordinal: int | None, task_id: str) -> str:
    """Return a sortable, human-readable raw-artifact directory name.

    ``task_id`` remains the stable machine identity.  The zero-padded ordinal
    makes a streamed run inspectable without looking up that id in cases.jsonl.
    Direct adapter callers without a stream ordinal use the task ID.
    """

    if ordinal is None:
        return task_id
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
        raise ValueError("run ordinal must be a positive integer or None")
    return f"{ordinal:04d}_{task_id}"


def _same_public_task(source_agent: dict[str, Any], public_payload: dict[str, Any]) -> bool:
    """Compare the source record with the public task after removing its schema wrapper."""

    normalized = copy.deepcopy(source_agent)
    normalized.pop("schema_version", None)
    return normalized == public_payload


@dataclass(frozen=True)
class NativeExecution:
    """A source-native row and its protected raw interaction."""

    row: dict[str, Any]
    interaction: dict[str, Any]


class AgentDoGGuardAdapter:
    """Adapt AgentDoG verdicts to the generic Guard protocol.

    The adapter only obtains a verdict. Whether a verdict leads to an
    annotation, action transformation, or withholding is decided by the
    configured Processor graph, not hard-coded in this adapter.
    """

    def __init__(self, diagnoser: Any, *, call_store: TraceStore | None = None) -> None:
        self.diagnoser = diagnoser
        self.call_store = call_store

    def with_call_store(self, call_store: TraceStore) -> "AgentDoGGuardAdapter":
        """Return a case-scoped adapter that shares the configured diagnoser."""

        return AgentDoGGuardAdapter(self.diagnoser, call_store=call_store)

    def diagnose(self, request: Any) -> GuardVerdict:
        from trajectory_schema import ToolCallRecord

        action = request.action
        tool_call = None
        if isinstance(action, CandidateAction) and action.tool_call is not None:
            tool_call = ToolCallRecord(
                tool_name=action.tool_call.name,
                tool_args=copy.deepcopy(action.tool_call.arguments),
                tool_call_id=action.tool_call.call_id,
            )
        user_request = next(
            (
                str(message.get("content", ""))
                for message in request.messages
                if isinstance(message, dict) and message.get("role") == "user"
            ),
            "",
        )
        diagnose_kwargs = {
            "user_request": user_request,
            "steps_so_far": [],
            "current_tool_call": tool_call,
            "hook": request.hook.value,
            "context_messages": list(request.messages),
            "available_tools": list(request.available_tools),
            "trusted_harness_context": getattr(request, "trusted_harness_context", None),
            "latest_event": copy.deepcopy(getattr(request, "latest_event", None)),
        }
        if self.call_store is not None:
            latest_event = getattr(request, "latest_event", None)
            latest_event_id = latest_event.get("event_id") if isinstance(latest_event, dict) else None
            action_id = (
                action.action_id
                if isinstance(action, CandidateAction)
                else str(latest_event_id or f"{request.hook.value}:unknown")
            )
            diagnose_kwargs.update(
                {
                    "call_store": self.call_store,
                    "call_context": {
                        "guard_call_id": f"{request.task_id}:{action_id}",
                        "task_id": request.task_id,
                        "action_id": action_id,
                        "hook": request.hook.value,
                        "subject_type": str(getattr(request, "subject_type", "unknown")),
                        "source": str(getattr(request, "source", "unknown")),
                    },
                }
            )
        diagnosis = self.diagnoser.diagnose(
            **diagnose_kwargs,
        )
        return GuardVerdict(
            safety=diagnosis.safety,
            risk_type=diagnosis.risk_type,
            risk_level=diagnosis.risk_level,
            reason=diagnosis.reason,
            evidence=tuple(str(item) for item in diagnosis.metadata.get("evidence", []) if isinstance(item, str)),
            availability="available" if diagnosis.safety != "unknown" else "parse_error",
            metadata={
                "parser": diagnosis.metadata.get("parser"),
                "agentdog_variant": diagnosis.metadata.get("agentdog_variant"),
                "guard_input": copy.deepcopy(diagnosis.metadata.get("guard_input") or {}),
            },
        )


def source_outcome(case: SafeCoEvoCase, row: dict[str, Any]) -> TaskOutcome:
    """Translate only source-native oracle values into the common outcome type."""

    source = case.private.source_benchmark
    status = str(row.get("status", "failed"))
    if source == "asb":
        attacked = bool(case.private.record.get("attack_schedule"))
        return TaskOutcome(
            status=status,
            unsafe_outcome=_bool_or_none(row.get("attack_success")) if attacked else None,
            legitimate_task_success=_bool_or_none(row.get("original_task_success")),
            verifier_name="ASB native environment oracle",
            verified=True,
            metadata={
                "attacked_variant": attacked,
                "official_signals": {
                    "attack_success": _bool_or_none(row.get("attack_success")) if attacked else None,
                    "task_success": _bool_or_none(row.get("original_task_success")),
                    "refusal": _bool_or_none(row.get("refused")),
                    "memory_found": _bool_or_none(row.get("memory_found")),
                },
            },
        )
    if source == "agent_safetybench":
        judge_parse_ok = row.get("agent_safetybench_judge_parse_ok") is True
        judge_label = row.get("agent_safetybench_judge_label")
        judge_unsafe = _bool_or_none(row.get("agent_safetybench_judge_unsafe"))
        task_completed = _bool_or_none(row.get("agent_safetybench_task_completed"))
        main_available = judge_parse_ok and judge_label in {0, 1, 2} and judge_unsafe is not None and task_completed is not None
        return TaskOutcome(
            status=status,
            unsafe_outcome=judge_unsafe if main_available else None,
            legitimate_task_success=task_completed if main_available else None,
            verifier_name="Agent-SafetyBench full-trajectory judge" if main_available else "Agent-SafetyBench pending full-trajectory judge",
            verified=main_available,
            metadata={
                "official_signals": {
                    "judge_label": row.get("agent_safetybench_judge_label"),
                    "judge_safe": row.get("agent_safetybench_judge_safe"),
                    "judge_unsafe": judge_unsafe,
                    "judge_partial_unsafe": row.get("agent_safetybench_judge_partial_unsafe"),
                    "judge_full_unsafe": row.get("agent_safetybench_judge_full_unsafe"),
                    "task_completed": task_completed,
                    "completion_label": row.get("agent_safetybench_completion_label"),
                    "refusal": row.get("agent_safetybench_refusal"),
                    "judge_parse_ok": judge_parse_ok,
                    "judge_model": row.get("agent_safetybench_main_judge_model"),
                    "fulfillable": row.get("fulfillable"),
                    "tool_call_count": row.get("tool_call_count"),
                    "protocol_error_count": row.get("protocol_error_count"),
                    # Judge rationale for the Evolution Agent: the parsed
                    # verdict labels alone force it to guess failure
                    # mechanisms.  Evidence lists are trimmed defensively.
                    "judge_reasoning": (str(row.get("agent_safetybench_judge_reasoning"))[:4000] or None) if row.get("agent_safetybench_judge_reasoning") else None,
                    "unsafe_evidence": [str(x)[:300] for x in (row.get("agent_safetybench_judge_unsafe_evidence") or [])][:6],
                    "completion_evidence": [str(x)[:300] for x in (row.get("agent_safetybench_judge_completion_evidence") or [])][:6],
                }
            },
        )

    raise ValueError(f"No SafeCoEvo outcome adapter for source {source!r}")


class LiveAgentSecurityBenchExecutor:
    """Drive ASB's released simulated tool/oracle semantics with a live Harness.

    ASB injects some source observations after a normal tool result and may
    schedule initial PoT/MP context.  The adapter preserves both delivery
    points through ``HarnessSession`` rather than converting them into a
    separate intervention.
    """

    source_adapter = "asb_live_harness"

    def __init__(
        self,
        *,
        dataset_dir: Path,
        curated_path: Path | None = None,
        model: str = "",
        base_url: str = "",
        max_tokens: int = 4096,
        timeout_seconds: float = 180.0,
        target_request_retries: int = 3,
        target_retry_sleep_seconds: float = 5.0,
        target_fallback_model: str = "",
        target_fallback_base_url: str = "",
        target_fallback_api_key: str = "",
        target_fallback_request_retries: int = 2,
    ) -> None:
        from benchmark_runtime import load_cases

        source_records = curated_path if curated_path is not None else dataset_dir / "records.jsonl"
        self.source_cases = {item.task_id: item for item in load_cases(dataset_dir, source_records)}
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.target_request_retries = target_request_retries
        self.target_retry_sleep_seconds = target_retry_sleep_seconds
        self.target_fallback_model = target_fallback_model
        self.target_fallback_base_url = target_fallback_base_url
        self.target_fallback_api_key = target_fallback_api_key
        self.target_fallback_request_retries = target_fallback_request_retries

    def public_task(self, case: SafeCoEvoCase) -> PublicTask:
        if case.private.source_benchmark != "asb":
            raise ValueError(f"Live ASB adapter received {case.private.source_benchmark}")
        source_case = self.source_cases.get(case.public.task_id)
        if source_case is None:
            raise KeyError(f"No source record for {case.public.task_id}")
        if not _same_public_task(source_case.agent, case.public.agent_payload()):
            raise ValueError(f"Public task drift for {case.public.task_id}; refusing mixed-version execution")
        from agent_security_bench_executor import public_task_with_source_context

        return public_task_with_source_context(source_case)

    def run(self, case: SafeCoEvoCase, session: HarnessSession) -> NativeExecution:
        if case.private.source_benchmark != "asb":
            raise ValueError(f"Live ASB adapter received {case.private.source_benchmark}")
        source_case = self.source_cases.get(case.public.task_id)
        if source_case is None:
            raise KeyError(f"No source record for {case.public.task_id}")
        from agent_security_bench_executor import run_case

        row, interaction = run_case(
            source_case,
            model=self.model,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
            target_request_retries=self.target_request_retries,
            target_retry_sleep_seconds=self.target_retry_sleep_seconds,
            target_fallback_model=self.target_fallback_model,
            target_fallback_base_url=self.target_fallback_base_url,
            target_fallback_api_key=self.target_fallback_api_key,
            target_fallback_request_retries=self.target_fallback_request_retries,
            harness_session=session,
        )
        return NativeExecution(row=row, interaction=interaction)


class LiveAgentSafetyBenchExecutor:
    """Drive Agent-SafetyBench environments through the live Harness loop.

    The released environment has no in-environment success oracle. Safety is
    scored by the full-trajectory judge after the episode, so this adapter
    records the complete interaction and leaves ``source_outcome`` pending
    until the runner applies that per-case judgment.
    """

    source_adapter = "agent_safetybench_live_harness"

    def __init__(
        self,
        *,
        dataset_dir: Path,
        curated_path: Path | None = None,
        model: str = "",
        base_url: str = "",
        max_tokens: int = 4096,
        timeout_seconds: float = 180.0,
        target_request_retries: int = 3,
        target_retry_sleep_seconds: float = 5.0,
        target_fallback_model: str = "",
        target_fallback_base_url: str = "",
        target_fallback_api_key: str = "",
        target_fallback_request_retries: int = 2,
    ) -> None:
        from benchmark_runtime import load_cases

        source_records = curated_path if curated_path is not None else dataset_dir / "records.jsonl"
        self.source_cases = {item.task_id: item for item in load_cases(dataset_dir, source_records)}
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.target_request_retries = target_request_retries
        self.target_retry_sleep_seconds = target_retry_sleep_seconds
        self.target_fallback_model = target_fallback_model
        self.target_fallback_base_url = target_fallback_base_url
        self.target_fallback_api_key = target_fallback_api_key
        self.target_fallback_request_retries = target_fallback_request_retries

    def run(self, case: SafeCoEvoCase, session: HarnessSession) -> NativeExecution:
        if case.private.source_benchmark != "agent_safetybench":
            raise ValueError(f"Live Agent-SafetyBench adapter received {case.private.source_benchmark}")
        source_case = self.source_cases.get(case.public.task_id)
        if source_case is None:
            raise KeyError(f"No source record for {case.public.task_id}")
        if not _same_public_task(source_case.agent, case.public.agent_payload()):
            raise ValueError(f"Public task drift for {case.public.task_id}; refusing mixed-version execution")
        from agent_safetybench_executor import run_case

        row, interaction = run_case(
            source_case,
            model=self.model,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
            target_request_retries=self.target_request_retries,
            target_retry_sleep_seconds=self.target_retry_sleep_seconds,
            target_fallback_model=self.target_fallback_model,
            target_fallback_base_url=self.target_fallback_base_url,
            target_fallback_api_key=self.target_fallback_api_key,
            target_fallback_request_retries=self.target_fallback_request_retries,
            harness_session=session,
        )
        return NativeExecution(row=row, interaction=interaction)




def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None
