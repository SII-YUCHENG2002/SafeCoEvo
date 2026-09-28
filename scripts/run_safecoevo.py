#!/usr/bin/env python3
"""Run the ASB and Agent-SafetyBench online SafeCoEvo stream.

Each target action passes through the Processor graph and AgentDoG Guard.
Native benchmark feedback then informs artifact evolution for later cases.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from runtime_api_config import (
    load_runtime_api_config,
    redacted_runtime_api_config_summary,
)  # noqa: E402


def _runtime_api_config_from_cli() -> Path | None:
    """Read the config path before importing components that consume its defaults."""

    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--runtime-api-config", type=Path)
    known, _unknown = bootstrap.parse_known_args()
    return known.runtime_api_config


# Load before argparse so every CLI default below comes from one fixed local
# configuration file rather than from inherited shell exports.
RUNTIME_API_CONFIG_PATH = load_runtime_api_config(_runtime_api_config_from_cli())

from agentdog_diagnoser import AgentDoGDiagnoser  # noqa: E402
from agentdog_preflight import require_agentdog_available  # noqa: E402
from model_clients import OpenAICompatibleClient  # noqa: E402
from safecoevo_runtime.artifact_patcher import hydrate_seed_memory_store, hydrate_seed_permission_store, load_seed_artifact_state  # noqa: E402
from safecoevo_runtime import (  # noqa: E402
    AgentDoGGuardAdapter,
    BatchEvolutionEpisode,
    LiveAgentSecurityBenchExecutor,
    LiveAgentSafetyBenchExecutor,
    OnlineGraphEvolutionController,
    OnlineEvolutionState,
    OpenAICompatibleEvolutionAgent,
    StaticEvolutionAgent,
    advance_graph_version,
    build_harness_from_graph,
    evolution_policy,
    initial_processor_graph,
    load_safecoevo_dataset,
    processor_graph_from_dict,
    SkillRegistry,
)
from safecoevo_runtime.execution import run_case_directory_name, source_outcome  # noqa: E402
from safecoevo_runtime.agent_safetybench_judge import (  # noqa: E402
    AgentSafetyBenchJudgeConfig,
    DEFAULT_JUDGE_PROMPT_PROFILE,
    JUDGE_PROMPT_PROFILES,
    build_full_trajectory_judge_record,
    judge_prompt_version,
    judge_agent_safetybench_record,
)
from safecoevo_runtime.memory_review_agent import make_memory_reviewer  # noqa: E402
from safecoevo_runtime.memory_review_runtime import (  # noqa: E402
    add_memory_review_arguments, attach_memory_review, check_review_resume, configure_memory_review,
)
from safecoevo_runtime.evolution_graph import EvolutionDeployment  # noqa: E402
from safecoevo_runtime.online_feedback import HarnessDelta, OfficialEpisodeFeedback  # noqa: E402
from safecoevo_runtime.skills import load_seed_skill_registry  # noqa: E402
from safecoevo_runtime.skill_consolidation import (  # noqa: E402
    last_consolidation_from_ledger,
    run_consolidation_round,
)
from safecoevo_runtime.skill_consolidation_journal import configure_consolidation_evidence_aware, execution_parameters  # noqa: E402
from safecoevo_runtime.trace import JsonTraceStore, load_trace_records  # noqa: E402


LIVE_SOURCES = {"asb", "agent_safetybench"}
GUARD_HOOK_POLICY = ("before_model", "before_tool", "after_tool")


def _configured_env(name: str) -> str:
    """Ignore inherited API routing when no JSON was selected."""
    return os.environ.get(name, "") if RUNTIME_API_CONFIG_PATH is not None else ""


def _target_cli_defaults() -> dict[str, str]:
    """Resolve Target routing after the selected runtime JSON populates env."""
    return {
        "model": _configured_env("SAFECOEVO_TARGET_MODEL"),
        "base_url": _configured_env("SAFECOEVO_TARGET_BASE_URL"),
        "fallback_model": _configured_env("SAFECOEVO_TARGET_FALLBACK_MODEL"),
        "fallback_base_url": _configured_env("SAFECOEVO_TARGET_FALLBACK_BASE_URL"),
        "fallback_api_key": _configured_env("SAFECOEVO_TARGET_FALLBACK_API_KEY"),
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _append_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n" for row in rows), encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _metric_fraction(numerator: int, denominator: int) -> dict[str, object]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _cumulative_overall_core_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build one cumulative UOR/TSR/SUCR snapshot per recorded case.

    Each metric uses its own availability-gated denominator. Runtime failures
    and unassessed outcomes remain visible in processed/failed counts without
    being silently treated as safe, unsafe, successful, or unsuccessful.
    """

    unsafe_count = unsafe_scorable = 0
    task_success_count = task_scorable = 0
    safe_useful_count = jointly_scorable = 0
    failed_count = 0
    snapshots: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        if row.get("status") == "failed":
            failed_count += 1
        metrics = row.get("unified_metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        unsafe_available = metrics.get("unsafe_outcome_available") is True
        task_available = metrics.get("legitimate_task_success_available") is True
        unsafe = metrics.get("unsafe_outcome")
        task_success = metrics.get("legitimate_task_success")
        if unsafe_available and isinstance(unsafe, bool):
            unsafe_scorable += 1
            unsafe_count += int(unsafe)
        if task_available and isinstance(task_success, bool):
            task_scorable += 1
            task_success_count += int(task_success)
        if unsafe_available and task_available and isinstance(unsafe, bool) and isinstance(task_success, bool):
            jointly_scorable += 1
            safe_useful_count += int((not unsafe) and task_success)
        snapshots.append(
            {
                "schema_version": 1,
                "ordinal": row.get("ordinal", index),
                "task_id": row.get("task_id"),
                "processed_rows": index,
                "failed_count": failed_count,
                "uor": _metric_fraction(unsafe_count, unsafe_scorable),
                "tsr": _metric_fraction(task_success_count, task_scorable),
                "sucr": _metric_fraction(safe_useful_count, jointly_scorable),
            }
        )
    return snapshots


def _empty_group_core_metrics() -> dict[str, object]:
    return {
        "processed_rows": 0,
        "failed_count": 0,
        "uor": _metric_fraction(0, 0),
        "tsr": _metric_fraction(0, 0),
        "sucr": _metric_fraction(0, 0),
    }


def cumulative_core_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Add source-specific reporting without changing overall metric semantics.

    Replay the same availability-gated accumulator for each group. Unknown or
    missing sources remain in overall metrics only; no task-order inference is
    used. Rebuilding from case rows keeps resumed group metrics consistent.
    """
    snapshots = _cumulative_overall_core_metrics(rows)
    for group, source in (("safety", "agent_safetybench"), ("security", "asb")):
        group_snapshots = iter(_cumulative_overall_core_metrics(
            [row for row in rows if row.get("source") == source]
        ))
        current = _empty_group_core_metrics()
        for row, snapshot in zip(rows, snapshots):
            if row.get("source") == source:
                grouped = next(group_snapshots)
                current = {key: grouped[key] for key in current}
            snapshot.setdefault("by_group", {})[group] = current
    return snapshots


def _persist_core_metrics(out_dir: Path, rows: list[dict[str, object]]) -> dict[str, object] | None:
    """Atomically rebuild progress/latest metrics so resume cannot duplicate rows."""

    snapshots = cumulative_core_metrics(rows)
    _append_jsonl(out_dir / "metrics_progress.jsonl", snapshots)
    latest = snapshots[-1] if snapshots else None
    _write_json(
        out_dir / "metrics_latest.json",
        latest or {
            "schema_version": 1,
            "ordinal": None,
            "task_id": None,
            "processed_rows": 0,
            "failed_count": 0,
            "uor": _metric_fraction(0, 0),
            "tsr": _metric_fraction(0, 0),
            "sucr": _metric_fraction(0, 0),
            "by_group": {
                "safety": _empty_group_core_metrics(),
                "security": _empty_group_core_metrics(),
            },
        },
    )
    return latest


def _attach_latest_core_metrics(rows: list[dict[str, object]], row: dict[str, object]) -> dict[str, object]:
    snapshot = cumulative_core_metrics([*rows, row])[-1]
    row["cumulative_core_metrics"] = snapshot
    return snapshot


def _metrics_log_event(snapshot: dict[str, object]) -> dict[str, object]:
    return {
        "event": "safecoevo_online_core_metrics",
        **snapshot,
    }


def _drop_trailing_failed_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Retry only a failed suffix so online evolution order remains valid."""

    cut = len(rows)
    while cut > 0 and rows[cut - 1].get("status") == "failed":
        cut -= 1
    return rows[:cut]


def load_task_manifest(path: Path, *, dataset: Path) -> tuple[str, ...]:
    """Load an ordered, reproducible online curriculum from a JSON manifest."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        declared_dataset = value.get("dataset")
        if declared_dataset is not None and Path(str(declared_dataset)).resolve() != dataset:
            raise ValueError(
                f"Task manifest dataset does not match --dataset: {declared_dataset} != {dataset}"
            )
        task_ids = value.get("task_ids")
    else:
        task_ids = value
    if not isinstance(task_ids, list) or not task_ids:
        raise ValueError("Task manifest must contain a non-empty task_ids list")
    if not all(isinstance(task_id, str) and task_id.strip() for task_id in task_ids):
        raise ValueError("Task manifest task_ids must be non-empty strings")
    ordered = tuple(task_id.strip() for task_id in task_ids)
    if len(ordered) != len(set(ordered)):
        raise ValueError("Task manifest must not contain duplicate task IDs")
    return ordered


def _restore_stream(
    *,
    out_dir: Path,
    cases: list[object],
    initial_version: str,
    resume: bool,
    retry_failed: bool,
    skill_config: dict[str, object],
    guard_action_mode: str,
    seed_graph: Path | None = None,
    seed_state: Path | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], OnlineEvolutionState, object, str]:
    """Restore an exact task-prefix checkpoint or initialize a new stream."""

    rows = _read_jsonl(out_dir / "cases.jsonl") if resume else []
    original_observed_ids = [row.get("task_id") for row in rows]
    if rows and retry_failed:
        trimmed = _drop_trailing_failed_rows(rows)
        if len(trimmed) == len(rows):
            raise ValueError("--retry-failed requires failed rows at the end of cases.jsonl")
        rows = trimmed
    evolution_rows = _read_jsonl(out_dir / "evolution_log.jsonl") if resume else []
    expected_ids = [case.public.task_id for case in cases]  # type: ignore[attr-defined]
    observed_ids = [row.get("task_id") for row in rows]
    if any(not isinstance(task_id, str) for task_id in observed_ids):
        raise ValueError("Existing cases.jsonl contains a row without task_id")
    if len(observed_ids) != len(set(observed_ids)) or observed_ids != expected_ids[:len(observed_ids)]:
        raise ValueError("Existing run results are not a unique prefix of this fixed online stream")
    if len(evolution_rows) > sum(isinstance(row.get("official_feedback"), dict) for row in rows):
        raise ValueError("Existing evolution_log.jsonl has more entries than completed feedback episodes")
    if not rows:
        if seed_state is not None:
            state_value = json.loads(seed_state.read_text(encoding="utf-8"))
            if not isinstance(state_value, dict):
                raise ValueError("Seed online state must be a JSON object")
            state = OnlineEvolutionState.from_dict(state_value)
            if state.initial_version != initial_version:
                raise ValueError(
                    f"Seed online state version {state.initial_version} does not match --initial-version {initial_version}"
                )
            if state.completed_episodes != 0 or state.feedback_history or state.deltas:
                raise ValueError("Seed online state must start with zero completed episodes")
        else:
            state = OnlineEvolutionState(initial_version=initial_version)
        if seed_graph is not None:
            graph_value = json.loads(seed_graph.read_text(encoding="utf-8"))
            if not isinstance(graph_value, dict):
                raise ValueError("Seed Processor graph must be a JSON object")
            graph = processor_graph_from_dict(graph_value)
            if graph.version != initial_version:
                raise ValueError(
                    f"Seed Processor graph version {graph.version} does not match --initial-version {initial_version}"
                )
        else:
            graph = initial_processor_graph(
                version=initial_version,
                memory_enabled=True,
                guard_enabled=True,
                guard_action_mode=guard_action_mode,
                skill_enabled=bool(skill_config["enabled"]),
                skill_auto_inject=bool(skill_config["auto_inject"]),
                skill_load_tool=bool(skill_config["load_tool"]),
                skill_top_k=int(skill_config["top_k"]),
                skill_max_active=int(skill_config["max_active"]),
            )
        return (
            rows,
            evolution_rows,
            state,
            graph,
            "fresh",
        )

    checkpoint = out_dir / "online_state.json"
    if checkpoint.exists():
        persisted = json.loads(checkpoint.read_text(encoding="utf-8"))
        if not isinstance(persisted, dict):
            raise ValueError("online_state.json must be an object")
        completed_ids = persisted.get("completed_task_ids")
        if completed_ids != observed_ids and not (retry_failed and completed_ids == original_observed_ids):
            raise ValueError("online_state.json does not match recorded task prefix")
        state_data = persisted.get("state")
        if not isinstance(state_data, dict):
            raise ValueError("online_state.json lacks state")
        state = OnlineEvolutionState.from_dict(state_data)
        restored_from = "checkpoint"
    else:
        raise ValueError("Cannot resume: online_state.json is missing; use a fresh --out-dir")
    completed_feedback = sum(isinstance(row.get("official_feedback"), dict) for row in rows)
    if state.completed_episodes != completed_feedback:
        raise ValueError("Restored state does not match completed official-feedback episodes")
    graph_path = out_dir / "evolution" / "versions" / state.version / "graph.json"
    if not graph_path.is_file():
        raise ValueError(f"Missing persisted active graph for {state.version}: {graph_path}")
    graph_value = json.loads(graph_path.read_text(encoding="utf-8"))
    if not isinstance(graph_value, dict):
        raise ValueError("Persisted active graph must be an object")
    graph = processor_graph_from_dict(graph_value)
    if _version_index(graph.version) > _version_index(state.version):
        raise ValueError("Persisted active graph version is ahead of restored stream state")
    return rows, evolution_rows, state, graph, restored_from


def _version_index(version: str) -> int:
    suffix = version[1:] if version.startswith("H") else version
    if not suffix:
        return 0
    try:
        return int(suffix.split(".")[-1])
    except ValueError:
        return 0


def _pending_batch_from_rows(
    *,
    rows: list[dict[str, object]],
    cases_by_id: dict[str, object],
    batch_size: int,
) -> list[dict[str, object]]:
    """Recover completed cases since the latest real deployment for resume."""

    if batch_size <= 1:
        return []
    pending: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row.get("official_feedback"), dict) or not isinstance(row.get("harness_delta"), dict):
            pending = []
            continue
        deployment = row.get("processor_graph_deployment")
        status = deployment.get("status") if isinstance(deployment, dict) else None
        if status == "deferred_batch":
            task_id = row.get("task_id")
            if isinstance(task_id, str) and task_id in cases_by_id:
                pending.append(row)
            continue
        pending = []
    return pending[-batch_size:]


def _write_checkpoint(
    *,
    out_dir: Path,
    state: OnlineEvolutionState,
    rows: list[dict[str, object]],
    graph_hash: str,
    skill_registry: SkillRegistry,
) -> None:
    _write_json(
        out_dir / "online_state.json",
        {
            "schema_version": 1,
            "completed_task_ids": [row["task_id"] for row in rows],
            "state": state.to_dict(),
            "processor_graph_hash": graph_hash,
            "skill_registry_version": skill_registry.version,
            "skill_registry_hash": skill_registry.registry_hash,
        },
    )


def _skill_runtime_config(args: argparse.Namespace) -> dict[str, object]:
    """Return the prompt-free runtime contract for the versioned skill set."""

    return {
        "enabled": bool(args.skill_runtime),
        "auto_inject": bool(args.skill_auto_inject),
        "load_tool": bool(args.skill_load_tool),
        "top_k": args.skill_top_k,
        "max_active": args.skill_max_active,
        "retrieval_mode": "lexical",
        "evolution_interval": args.skill_evolution_interval,
        "registry_input": str(args.skill_registry.resolve()) if args.skill_registry is not None else None,
    }


def _restore_skill_registry(*, out_dir: Path, resume: bool, initial: Path | None) -> tuple[SkillRegistry, str]:
    """Restore the exact active version used by the next online episode."""

    checkpoint = out_dir / "online_state.json"
    if resume and checkpoint.is_file():
        value = json.loads(checkpoint.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("online_state.json must be an object")
        version = value.get("skill_registry_version")
        if isinstance(version, str):
            path = out_dir / "skills" / "versions" / version / "registry.json"
            if not path.is_file():
                raise ValueError(f"Missing persisted active Skill registry: {path}")
            registry = SkillRegistry.load(path)
            if value.get("skill_registry_hash") not in {None, registry.registry_hash}:
                raise ValueError("Persisted Skill registry hash differs from online_state.json")
            return registry, "checkpoint"
    if initial is not None:
        return SkillRegistry.load(initial.resolve()), "initial_registry"
    seed = ROOT / "artifacts" / "skills"
    if seed.is_dir() and (seed / "registry.json").is_file():
        return load_seed_skill_registry(seed), "seed_artifacts"
    return SkillRegistry(), "empty_registry"


def _persist_skill_registry(*, out_dir: Path, registry: SkillRegistry) -> Path:
    path = out_dir / "skills" / "versions" / registry.version / "registry.json"
    registry.write(path)
    return path


def _advance_consolidation_evidence_aware(runtime, *, args, rows, state, evolver, registry):
    if runtime is None:
        return registry
    def publish(successor):
        _persist_skill_registry(out_dir=args.out_dir, registry=successor)
        _write_checkpoint(out_dir=args.out_dir, state=state, rows=rows,
                          graph_hash=evolver.graph.graph_hash, skill_registry=successor)
    previous = len(runtime.data['rounds'])
    registry = runtime.advance(rows=rows, memory_items=[{
        'memory_id': item.memory_id, 'content': item.content, 'tags': list(item.tags),
        'verified': bool(item.verified), 'evidence_trace_hashes': list(item.evidence_trace_hashes),
    } for item in state.memory_store.items], registry=registry, agent=evolver.agent, persist_registry=publish)
    evolver.skill_registry = registry
    if len(runtime.data['rounds']) > previous:
        print(json.dumps({'event': 'skill_consolidation_evidence_aware', 'ordinal': len(rows),
                          **runtime.last_result}, ensure_ascii=False), flush=True)
    return registry


def _batch_episode_from_row(
    *,
    row: dict[str, object],
    case: object,
    graph,
) -> BatchEvolutionEpisode:
    """Rehydrate a deferred completed episode with exact trace records."""

    feedback_payload = row.get("official_feedback")
    delta_payload = row.get("harness_delta")
    trace_path = row.get("harness_trace_path")
    if not isinstance(feedback_payload, dict) or not isinstance(delta_payload, dict) or not isinstance(trace_path, str):
        raise ValueError("Deferred batch row lacks feedback, delta, or trace path")
    trace_records = load_trace_records(Path(trace_path))
    session = _RehydratedSession(
        task=case.public,  # type: ignore[attr-defined]
        graph_version=graph.version,
        trace_hashes=[
            str(record.get("entry_hash"))
            for record in trace_records
            if isinstance(record, dict) and record.get("entry_hash")
        ],
    )
    feedback = _feedback_from_payload(feedback_payload)
    delta = HarnessDelta(
        parent_version=str(delta_payload["parent_version"]),
        next_version=str(delta_payload["next_version"]),
        task_id=str(delta_payload["task_id"]),
        change_type=str(delta_payload["change_type"]),
        memory_id=str(delta_payload["memory_id"]) if delta_payload.get("memory_id") is not None else None,
        rationale=str(delta_payload["rationale"]),
        evidence_trace_hashes=tuple(str(item) for item in delta_payload.get("evidence_trace_hashes", []) if item),
    )
    session.state.annotations["trace_hashes"] = list(delta.evidence_trace_hashes)
    return BatchEvolutionEpisode(
        case=case,  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
        feedback=feedback,
        trace_records=trace_records,
        next_version=delta.next_version,
    )


class _RehydratedSession:
    """Minimal completed-session surface needed for batch evidence resume."""

    def __init__(self, *, task: object, graph_version: str, trace_hashes: list[str]) -> None:
        from types import SimpleNamespace

        self.task = task
        self.harness = SimpleNamespace(config=SimpleNamespace(version=graph_version))
        self.state = SimpleNamespace(
            annotations={"trace_hashes": trace_hashes},
            evolution_signals=[],
            evolution_evidence=[],
        )




def _unified_metrics_from_feedback(feedback: OfficialEpisodeFeedback) -> dict[str, object]:
    """Convert official feedback into core safety and utility metrics."""

    unsafe = {"passed": False, "failed": True}.get(feedback.safety)
    task_success = {"passed": True, "failed": False}.get(feedback.goal)
    unsafe_available = isinstance(unsafe, bool)
    task_available = isinstance(task_success, bool)
    return {
        "unsafe_outcome": unsafe if unsafe_available else None,
        "unsafe_outcome_available": unsafe_available,
        "legitimate_task_success": task_success if task_available else None,
        "legitimate_task_success_available": task_available,
        "safe_and_useful_completion": (not unsafe) and task_success if unsafe_available and task_available else None,
        "safe_and_useful_completion_available": unsafe_available and task_available,
    }


def _safe_native_result_fields(source: str, row: dict[str, object]) -> dict[str, object]:
    """Keep prompt-free native evaluator fields for later audit and analysis."""

    keys_by_source = {
        "asb": ("attack_success", "original_task_success", "refused", "memory_found", "tool_call_count"),
        "agent_safetybench": (
            "fulfillable",
            "protocol_error_count",
            "tool_call_count",
            "agent_safetybench_main_judge",
            "agent_safetybench_main_judge_model",
            "agent_safetybench_judge_parse_ok",
            "agent_safetybench_judge_label",
            "agent_safetybench_judge_safe",
            "agent_safetybench_judge_unsafe",
            "agent_safetybench_judge_partial_unsafe",
            "agent_safetybench_judge_full_unsafe",
            "agent_safetybench_task_completed",
            "agent_safetybench_completion_label",
            "agent_safetybench_refusal",
            "agent_safetybench_judge_error",
            "agent_safetybench_judge_endpoint",
            "agent_safetybench_judge_base_url",
            "agent_safetybench_judge_attempts",
        ),
    }
    keys = keys_by_source.get(source, ())
    return {key: row.get(key) for key in keys if key in row}


def _top_level_audit_fields(source: str, row: dict[str, object]) -> dict[str, object]:
    """Expose selected Agent-SafetyBench evaluator fields at top level."""

    if source != "agent_safetybench":
        return {}
    keys = (
        "agent_safetybench_main_judge",
        "agent_safetybench_main_judge_model",
        "agent_safetybench_judge_parse_ok",
        "agent_safetybench_judge_label",
        "agent_safetybench_judge_safe",
        "agent_safetybench_judge_unsafe",
        "agent_safetybench_judge_partial_unsafe",
        "agent_safetybench_judge_full_unsafe",
        "agent_safetybench_task_completed",
        "agent_safetybench_completion_label",
        "agent_safetybench_refusal",
        "agent_safetybench_judge_error",
        "agent_safetybench_judge_endpoint",
        "agent_safetybench_judge_base_url",
        "agent_safetybench_judge_attempts",
    )
    return {key: row.get(key) for key in keys if key in row}


def _feedback_from_payload(value: dict[str, object]) -> OfficialEpisodeFeedback:
    availability = value.get("availability")
    trace_summary = value.get("trace_summary")
    official_signals = value.get("official_signals")
    return OfficialEpisodeFeedback(
        task_id=str(value["task_id"]),
        harness_version_before=str(value["harness_version_before"]),
        safety=str(value["safety"]),  # type: ignore[arg-type]
        goal=str(value["goal"]),  # type: ignore[arg-type]
        evaluation_complete=bool(value["evaluation_complete"]),
        evaluator_name=str(value["evaluator_name"]) if value.get("evaluator_name") is not None else None,
        evaluator_kind=str(value["evaluator_kind"]),
        official_signals=dict(official_signals) if isinstance(official_signals, dict) else {},
        availability=dict(availability) if isinstance(availability, dict) else {},
        trace_summary=dict(trace_summary) if isinstance(trace_summary, dict) else {},
    )


def _deferred_batch_deployment(
    *,
    parent_version: str,
    next_version: str,
    task_id: str,
    batch_size: int,
    pending_count: int,
) -> EvolutionDeployment:
    return EvolutionDeployment(
        parent_version=parent_version,
        next_version=next_version,
        task_id=task_id,
        candidate_id=None,
        status="deferred_batch",
        reason=f"Evolution is deferred until the current batch reaches {batch_size} completed episodes.",
        validation={
            "accepted": False,
            "mode": "fixed_artifact_patch",
            "evolution_granularity": "batch",
            "batch_size": batch_size,
            "pending_episodes": pending_count,
        },
    )


def select_online_cases(
    cases: list[object],
    *,
    sources: set[str],
    task_ids: tuple[str, ...],
    max_cases: int,
) -> list[object]:
    """Select a deterministic online stream, preserving an explicit ID order.

    Episode order is experimental state: feedback from one episode can be
    retrieved in the next. A set-based filter would silently replace a caller's
    requested curriculum with dataset order, so explicit task IDs are ordered
    and duplicate IDs are rejected.
    """

    scoped = [
        case for case in cases
        if case.private.source_benchmark in sources  # type: ignore[attr-defined]
    ]
    if not task_ids:
        return scoped if max_cases <= 0 else scoped[:max_cases]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("--task-ids must not contain duplicates in an online stream")
    by_id = {case.public.task_id: case for case in scoped}  # type: ignore[attr-defined]
    missing = sorted(set(task_ids) - set(by_id))
    if missing:
        raise ValueError(
            "Requested task IDs are not in the selected live-source scope: "
            f"{missing}"
        )
    ordered = [by_id[task_id] for task_id in task_ids]
    return ordered if max_cases <= 0 else ordered[:max_cases]


def _make_guard(args: argparse.Namespace) -> AgentDoGGuardAdapter:
    require_agentdog_available(
        base_url=args.guard_base_url,
        model=args.guard_model,
        api_key_env=args.guard_api_key_env,
        event="safecoevo_online_guard_preflight",
        max_tokens=args.guard_max_tokens,
    )
    return AgentDoGGuardAdapter(
        AgentDoGDiagnoser(
            OpenAICompatibleClient(
                base_url=args.guard_base_url,
                model=args.guard_model,
                api_key_env=args.guard_api_key_env,
                temperature=0,
                max_tokens=args.guard_max_tokens,
                timeout_seconds=args.guard_timeout_seconds,
            )
        )
    )


def _guard_for_case(guard: AgentDoGGuardAdapter, case_dir: Path) -> AgentDoGGuardAdapter:
    """Bind exact Guard API-call auditing to one task artifact directory."""

    return guard.with_call_store(JsonTraceStore(case_dir / "guard_calls.json"))


def _memory_retrieval_config() -> dict[str, object]:
    """Build a secret-free retrieval contract for plans and checkpoints."""

    return {
        "mode": "lexical",
        "query_source": "public_user_messages_and_public_tool_names",
        "memory_document_source": "verified_generic_lesson_and_public_tags",
        "external_data_acknowledged": False,
    }


def main() -> int:
    target_defaults = _target_cli_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-api-config",
        type=Path,
        default=RUNTIME_API_CONFIG_PATH,
        help="Persistent JSON API configuration selected before runner imports (default: configs/runtime_api_config.json).",
    )
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Directory containing the supplied SafeCoEvo task stream.")
    parser.add_argument("--curated", type=Path, default=None,
                        help="Source records JSONL (default: <dataset>/records.jsonl).")
    parser.add_argument("--seed-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument(
        "--seed-graph",
        type=Path,
        default=None,
        help="Frozen Processor graph JSON to use for a fresh stream; must match --initial-version.",
    )
    parser.add_argument(
        "--seed-state",
        type=Path,
        default=None,
        help="Frozen OnlineEvolutionState JSON for a fresh stream; must have zero completed episodes and match --initial-version.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-failed", action="store_true", help="Drop a trailing failed suffix and retry it without changing the prior online state.")
    parser.add_argument("--sources", default="asb,agent_safetybench")
    parser.add_argument("--max-cases", type=int, default=0, help="0 = run the whole configured stream.")
    parser.add_argument("--task-ids", default="")
    parser.add_argument(
        "--task-manifest",
        type=Path,
        help="Ordered JSON task manifest. Mutually exclusive with --task-ids.",
    )
    # Endpoint identities and credentials are selected exclusively by the JSON.
    parser.set_defaults(
        model=target_defaults["model"],
        base_url=target_defaults["base_url"],
        target_fallback_model=target_defaults["fallback_model"],
        target_fallback_base_url=target_defaults["fallback_base_url"],
        target_fallback_api_key=target_defaults["fallback_api_key"],
        guard_base_url=_configured_env("AGENTDOG_BASE_URL"),
        guard_model=_configured_env("AGENTDOG_MODEL"),
        guard_api_key_env="AGENTDOG_API_KEY",
        agent_safetybench_judge_model=_configured_env("SAFECOEVO_AGENT_SAFETYBENCH_JUDGE_MODEL"),
        agent_safetybench_judge_base_url=_configured_env("SAFECOEVO_AGENT_SAFETYBENCH_JUDGE_BASE_URL"),
        agent_safetybench_judge_api_key_env="SAFECOEVO_JUDGE_API_KEY",
        agent_safetybench_judge_user_agent=_configured_env("SAFECOEVO_AGENT_SAFETYBENCH_JUDGE_USER_AGENT"),
        agent_safetybench_judge_fallback_model=_configured_env("SAFECOEVO_AGENT_SAFETYBENCH_JUDGE_FALLBACK_MODEL"),
        agent_safetybench_judge_fallback_base_url=_configured_env("SAFECOEVO_AGENT_SAFETYBENCH_JUDGE_FALLBACK_BASE_URL"),
        agent_safetybench_judge_fallback_api_key_env="SAFECOEVO_JUDGE_FALLBACK_API_KEY",
        agent_safetybench_judge_fallback_user_agent=_configured_env("SAFECOEVO_AGENT_SAFETYBENCH_JUDGE_FALLBACK_USER_AGENT"),
        evolution_model=_configured_env("SAFECOEVO_EVOLUTION_MODEL"),
        evolution_base_url=_configured_env("SAFECOEVO_EVOLUTION_BASE_URL"),
        evolution_api_key_env="SAFECOEVO_EVOLUTION_API_KEY",
        evolution_fallback_model=_configured_env("SAFECOEVO_EVOLUTION_FALLBACK_MODEL"),
        evolution_fallback_base_url=_configured_env("SAFECOEVO_EVOLUTION_FALLBACK_BASE_URL"),
        evolution_fallback_api_key=_configured_env("SAFECOEVO_EVOLUTION_FALLBACK_API_KEY"),
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--target-request-retries", type=int, default=3)
    parser.add_argument("--target-retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--target-fallback-request-retries", type=int, default=2)
    parser.add_argument("--guard-max-tokens", type=int, default=2048)
    parser.add_argument("--guard-timeout-seconds", type=int, default=180)
    parser.add_argument("--guard-action-mode", choices=("advisory", "enforcing"), default="enforcing", help="Native tool handling after an unsafe Guard verdict.")
    parser.add_argument(
        "--agent-safetybench-full-judge",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--agent-safetybench-judge-temperature", type=float, default=float(os.environ.get("SAFECOEVO_AGENT_SAFETYBENCH_JUDGE_TEMPERATURE", "0.0")))
    parser.add_argument("--agent-safetybench-judge-max-tokens", type=int, default=int(os.environ.get("SAFECOEVO_AGENT_SAFETYBENCH_JUDGE_MAX_TOKENS", "8192")))
    parser.add_argument("--agent-safetybench-judge-timeout-seconds", type=int, default=240)
    parser.add_argument("--agent-safetybench-judge-request-retries", type=int, default=2)
    parser.add_argument("--agent-safetybench-judge-retry-sleep-seconds", type=float, default=10.0)
    parser.add_argument("--agent-safetybench-judge-fallback-request-retries", type=int, default=int(os.environ.get("SAFECOEVO_AGENT_SAFETYBENCH_JUDGE_FALLBACK_REQUEST_RETRIES", "2")))
    parser.add_argument(
        "--agent-safetybench-judge-prompt-profile",
        choices=tuple(JUDGE_PROMPT_PROFILES),
        default=os.environ.get("SAFECOEVO_AGENT_SAFETYBENCH_JUDGE_PROMPT_PROFILE", DEFAULT_JUDGE_PROMPT_PROFILE),
        help="Agent-SafetyBench judge prompt profile used in result metadata.",
    )
    add_memory_review_arguments(parser)
    parser.add_argument(
        "--skill-runtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the versioned Skill catalog and internal LoadSkill runtime for online Harness episodes.",
    )
    parser.add_argument(
        "--skill-auto-inject",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inject top-matching Skill references only into the first target-model request of an episode.",
    )
    parser.add_argument(
        "--skill-load-tool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expose the Harness-internal read-only LoadSkill tool when the source task has tools.",
    )
    parser.add_argument("--skill-top-k", type=int, default=3)
    parser.add_argument("--skill-max-active", type=int, default=8)
    parser.add_argument(
        "--skill-registry",
        type=Path,
        help="Optional initial SkillRegistry JSON. The active copy is versioned under --out-dir/skills/.",
    )
    parser.add_argument(
        "--skill-evolution-interval",
        type=int,
        default=1,
        help="Number of completed episodes an Evolution Agent reviews before Skill candidate deployment (default: 1).",
    )
    parser.add_argument(
        "--skill-consolidation-interval",
        type=int,
        default=int(os.environ.get("SAFECOEVO_SKILL_CONSOLIDATION_INTERVAL", "0")),
        help="Run a Memory->Skill consolidation round every N episodes (0 disables; default 0).",
    )
    parser.add_argument('--skill-consolidation-implementation', choices=('standard', 'evidence_aware'), default='standard',
                        help='Use evidence-aware consolidation with durable periodic journaling.')
    parser.add_argument('--skill-consolidation-max-input-chars', type=int, default=250000,
                        help='Evidence-aware consolidation skips oversized inputs without truncation.')
    parser.add_argument(
        "--skill-consolidation-min-episodes",
        type=int,
        default=30,
        help="Never consolidate before this episode ordinal.",
    )
    parser.add_argument(
        "--skill-consolidation-min-new-lessons",
        type=int,
        default=10,
        help="Trigger an early consolidation when this many new lessons accumulated since the last round.",
    )
    parser.add_argument(
        "--evolution-batch-size",
        type=int,
        default=1,
        help="Run the Evolution Agent once after this many completed episodes; 1 keeps per-episode evolution.",
    )
    parser.add_argument("--initial-version", default="H0")
    parser.add_argument("--evolution-mode", choices=("model", "off"), default="model")
    parser.add_argument(
        "--evolution-max-tokens",
        type=int,
        default=16384,
        help="Maximum tokens for each Evolution Agent completion (default: 16384).",
    )
    parser.add_argument(
        "--evolution-timeout-seconds",
        type=float,
        default=240.0,
        help="Per-request client timeout; provider stream limits may still fail earlier (default: 240).",
    )
    parser.add_argument(
        "--evolution-request-retries",
        type=int,
        default=3,
        help="Safe primary retries for an unanswered model request; tools execute only after a response arrives (default: 3).",
    )
    parser.add_argument("--evolution-retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--evolution-fallback-request-retries", type=int, default=2)
    parser.add_argument(
        "--evolution-max-tool-rounds",
        type=int,
        default=100,
        help="Maximum read/write/validation tool rounds; source review uses one complete bundle call (default: 100).",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if RUNTIME_API_CONFIG_PATH is None and (args.execute or args.runtime_api_config is not None):
        parser.error("runtime API config file does not exist; select a valid --runtime-api-config JSON")
    if args.memory_review_max_input_chars < 1:
        parser.error("--memory-review-max-input-chars must be positive")
    if args.memory_review_mode == "apply" and args.evolution_mode != "model":
        parser.error("Memory review apply requires --evolution-mode model; use audit for a frozen run")
    if args.max_cases < 0:
        parser.error("--max-cases must be non-negative (0 = run the whole configured stream)")
    if args.guard_max_tokens < 1 or args.guard_timeout_seconds < 1:
        parser.error("Guard limits must be positive")
    if args.target_request_retries < 0 or args.target_fallback_request_retries < 0 or args.target_retry_sleep_seconds < 0:
        parser.error("Target model retry counts and sleep must be non-negative")
    if (
        args.evolution_max_tokens < 1
        or args.evolution_timeout_seconds <= 0
        or args.evolution_request_retries < 0
        or args.evolution_fallback_request_retries < 0
        or args.evolution_retry_sleep_seconds < 0
        or args.evolution_max_tool_rounds < 1
    ):
        parser.error("Evolution Agent token/timeout/tool limits must be positive and retry settings non-negative")
    if args.skill_top_k < 1 or args.skill_evolution_interval < 1 or args.evolution_batch_size < 1:
        parser.error("Skill top-k, skill evolution interval, and evolution batch size must be positive")
    if args.skill_max_active < 1:
        parser.error("Skill max-active must be at least 1")
    selected_policy = evolution_policy("fixed_artifact_patch")

    dataset = args.dataset.resolve()
    curated_path = (args.curated if args.curated is not None else dataset / "records.jsonl").resolve()
    selected_sources = {item.strip() for item in args.sources.split(",") if item.strip()}
    unsupported = selected_sources - LIVE_SOURCES
    if unsupported:
        parser.error(f"Unsupported sources: {sorted(unsupported)}")
    try:
        if args.task_manifest is not None and args.task_ids.strip():
            raise ValueError("--task-manifest and --task-ids are mutually exclusive")
        requested_ids = (
            load_task_manifest(args.task_manifest.resolve(), dataset=dataset)
            if args.task_manifest is not None
            else tuple(item.strip() for item in args.task_ids.split(",") if item.strip())
        )
        cases = select_online_cases(
            load_safecoevo_dataset(dataset),
            sources=selected_sources,
            task_ids=requested_ids,
            max_cases=args.max_cases,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not cases:
        parser.error("No cases selected")

    args.out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if args.resume and (args.out_dir / "online_state.json").is_file():
        checkpoint_header = json.loads((args.out_dir / "online_state.json").read_text(encoding="utf-8"))
        state_header = checkpoint_header.get("state") if isinstance(checkpoint_header, dict) else None
        if not isinstance(state_header, dict) or state_header.get("schema_version") != 3:
            parser.error("Unsupported online state schema; start a fresh output directory")
    committed_rows = _read_jsonl(args.out_dir / "cases.jsonl")
    skill_config = _skill_runtime_config(args)
    plan = {
        "schema_version": 5,
        "protocol": "official_feedback_fixed_artifact_evolution",
        "dataset": str(dataset),
        "curated": str(curated_path),
        "runtime_api_config": redacted_runtime_api_config_summary(),
        "target": {
            "model": args.model,
            "base_url": args.base_url,
            "temperature": 0.0,
            "thinking": False,
            "max_turns": sorted({int(case.public.runtime["max_turns"]) for case in cases}),
            "request_retries": args.target_request_retries,
            "retry_sleep_seconds": args.target_retry_sleep_seconds,
            "fallback": {
                "model": args.target_fallback_model,
                "base_url": args.target_fallback_base_url,
                "request_retries": args.target_fallback_request_retries,
                "api_key_source": "runtime_api_config",
            },
        },
        "guard": {
            "model": args.guard_model,
            "base_url": args.guard_base_url,
            "max_tokens": args.guard_max_tokens,
            "hook": "three_hook_guard",
            "hooks": list(GUARD_HOOK_POLICY),
        },
        "feedback": {
            "timing": "post_episode_official_feedback; evolution_after_each_episode_or_batch_boundary",
            "mode": "native_official_oracle",
            "agent_safetybench_post_scorer": {
                "main_judge_enabled": args.agent_safetybench_full_judge,
                "main_judge": "agent_safetybench_full_trajectory_judge",
                "main_judge_model": args.agent_safetybench_judge_model,
                "main_judge_base_url": args.agent_safetybench_judge_base_url,
                "main_judge_user_agent": args.agent_safetybench_judge_user_agent,
                "main_judge_temperature": args.agent_safetybench_judge_temperature,
                "main_judge_max_tokens": args.agent_safetybench_judge_max_tokens,
                "main_judge_prompt_profile": args.agent_safetybench_judge_prompt_profile,
                "main_judge_prompt_version": judge_prompt_version(args.agent_safetybench_judge_prompt_profile),
                "main_judge_fallback": {
                    "model": args.agent_safetybench_judge_fallback_model,
                    "base_url": args.agent_safetybench_judge_fallback_base_url,
                    "api_key_env": args.agent_safetybench_judge_fallback_api_key_env,
                    "user_agent": args.agent_safetybench_judge_fallback_user_agent,
                    "request_retries": args.agent_safetybench_judge_fallback_request_retries,
                },
            },
        },
        "memory_retrieval": _memory_retrieval_config(),
        "asb_deferred_memory": {"mode": "live_retrieval"},
        "skill_runtime": skill_config,
        "evolution": {
            "mode": args.evolution_mode,
            "agent_view_contract": "source_blind_evolution_view",
            "benchmark_provenance_available_to_evolution_agent": False,
            "policy": selected_policy.to_dict(),
            "agent_model": args.evolution_model if args.evolution_mode == "model" else None,
            "agent_base_url": args.evolution_base_url if args.evolution_mode == "model" else None,
            "max_tokens": args.evolution_max_tokens if args.evolution_mode == "model" else None,
            "timeout_seconds": args.evolution_timeout_seconds if args.evolution_mode == "model" else None,
            "request_retries": args.evolution_request_retries if args.evolution_mode == "model" else None,
            "retry_sleep_seconds": args.evolution_retry_sleep_seconds if args.evolution_mode == "model" else None,
            "fallback": (
                {
                    "model": args.evolution_fallback_model,
                    "base_url": args.evolution_fallback_base_url,
                    "request_retries": args.evolution_fallback_request_retries,
                    "api_key_source": "runtime_api_config",
                }
                if args.evolution_mode == "model"
                else None
            ),
            "max_tool_rounds": args.evolution_max_tool_rounds if args.evolution_mode == "model" else None,
            "allowed_file_updates": ["append_text", "replace_text", "append_jsonl", "update_jsonl", "replace_json", "json_patch"],
            "allowed_artifacts": [
                "prompt/target_system_addendum.md",
                "memory/validated_experience.jsonl",
                "skills/registry.json",
                "skills/<skill_id>/SKILL.md",
                "permission/permission_experience.jsonl",
                "guard/guard_policy.json",
            ],
            "workspace": "complete_source_blind_trace_plus_readonly_artifact_history_workspace",
            "deployment": "artifact_patch_validate_then_next_episode_only" if args.evolution_batch_size == 1 else "artifact_patch_validate_after_batch_then_next_episode_only",
            "batch_size": args.evolution_batch_size,
            "granularity": "per_episode" if args.evolution_batch_size == 1 else "batch",
            "processor_graph_evolution": False,
        },
        "initial_harness_version": args.initial_version,
        "task_manifest": str(args.task_manifest.resolve()) if args.task_manifest is not None else None,
        "selected_cases": [{"task_id": case.public.task_id, "source": case.private.source_benchmark} for case in cases],
        "resume": args.resume,
        "execute": args.execute,
    }
    # Freeze public execution configuration before any restored registry can be reconfigured/written.
    consolidation_evidence_aware = configure_consolidation_evidence_aware(args, committed_rows,
        execution_contract=execution_parameters(args) if args.skill_consolidation_implementation == 'evidence_aware' else None)
    if consolidation_evidence_aware is not None:
        plan['skill_consolidation'] = consolidation_evidence_aware.configuration
    existing_rows, evolution_rows, state, graph, restored_from = _restore_stream(
        out_dir=args.out_dir,
        cases=cases,
        initial_version=args.initial_version,
        resume=args.resume,
        retry_failed=args.retry_failed,
        skill_config=skill_config,
        guard_action_mode=args.guard_action_mode,
        seed_graph=args.seed_graph.resolve() if args.seed_graph is not None else None,
        seed_state=args.seed_state.resolve() if args.seed_state is not None else None,
    )
    cases_by_id = {case.public.task_id: case for case in cases}
    required_skill_nodes = {
        "skill_catalog",
        "skill_runtime",
        "progressive_skill_loader",
        "skill_tool_provider",
    }
    active_node_ids = {node.processor_id for node in graph.nodes}
    if args.skill_runtime and not required_skill_nodes <= active_node_ids:
        parser.error(
            "The Processor graph lacks required Skill Runtime nodes. Start a fresh --out-dir "
            "with a graph that includes those nodes."
        )
    state.configure_memory_retrieval(_memory_retrieval_config())
    memory_review = configure_memory_review(args)
    check_review_resume(memory_review.configuration, existing_rows)
    if memory_review.mode != "off":
        plan["memory_review"] = memory_review.configuration
    skill_registry, skill_registry_restored_from = _restore_skill_registry(
        out_dir=args.out_dir,
        resume=args.resume,
        initial=args.skill_registry,
    )
    if not args.resume and (args.out_dir / "cases.jsonl").exists():
        parser.error("--no-resume refuses to overwrite an existing online stream")
    if not state.artifact_state.prompt_addendum:
        state.artifact_state = load_seed_artifact_state(args.seed_root.resolve())
    seed_memory_loaded = 0
    seed_permission_loaded = 0
    if restored_from == "fresh" and args.seed_state is None:
        seed_memory_loaded = hydrate_seed_memory_store(args.seed_root.resolve(), state.memory_store)
        seed_permission_loaded = hydrate_seed_permission_store(args.seed_root.resolve(), state.permission_store)
    elif restored_from == "checkpoint":
        checkpoint_state = json.loads((args.out_dir / "online_state.json").read_text(encoding="utf-8"))["state"]
        if "permission_items" not in checkpoint_state:
            ledger_path = args.out_dir / "evolution" / "ledger" / "artifact_patch_ledger.jsonl"
            for entry in _read_jsonl(ledger_path):
                if any(update.get("path") == "permission/permission_experience.jsonl"
                       for update in entry.get("runtime_updates", []) if isinstance(update, dict)):
                    raise ValueError("Checkpoint does not contain deployed Permission experience; resume requires a checkpoint with permission_items")
            seed_permission_loaded = hydrate_seed_permission_store(args.seed_root.resolve(), state.permission_store)
    _persist_skill_registry(out_dir=args.out_dir, registry=skill_registry)
    pending_batch_rows = _pending_batch_from_rows(
        rows=existing_rows,
        cases_by_id=cases_by_id,
        batch_size=args.evolution_batch_size,
    )
    plan["resume_state"] = {
        "restored_from": restored_from,
        "recorded_cases": len(existing_rows),
        "active_harness_version": state.version,
        "active_artifact_version": state.artifact_state.version,
        "active_memory_count": len(state.memory_store.items),
        "active_permission_count": len(state.permission_store.items),
        "active_skill_registry_version": skill_registry.version,
        "seed_memory_loaded": seed_memory_loaded,
        "seed_permission_loaded": seed_permission_loaded,
        "active_skill_registry_hash": skill_registry.registry_hash,
        "skill_registry_restored_from": skill_registry_restored_from,
        "pending_batch_episodes": len(pending_batch_rows),
    }
    _write_json(args.out_dir / "run_plan.json", plan)
    print(json.dumps({"event": "safecoevo_online_plan", "selected": len(cases), "by_source": dict(sorted(Counter(case.private.source_benchmark for case in cases).items())), "recorded": len(existing_rows), "active_harness_version": state.version, "restored_from": restored_from, "execute": args.execute}, ensure_ascii=False), flush=True)
    if not args.execute:
        return 0

    if not os.environ.get("INF_API_KEY"):
        raise RuntimeError("Missing Target API key in selected runtime API config")
    guard = _make_guard(args)
    agent_safetybench_judge_config = AgentSafetyBenchJudgeConfig(
        model=args.agent_safetybench_judge_model,
        base_url=args.agent_safetybench_judge_base_url,
        api_key_env=args.agent_safetybench_judge_api_key_env,
        user_agent=args.agent_safetybench_judge_user_agent,
        max_tokens=args.agent_safetybench_judge_max_tokens,
        timeout_seconds=args.agent_safetybench_judge_timeout_seconds,
        temperature=args.agent_safetybench_judge_temperature,
        request_retries=args.agent_safetybench_judge_request_retries,
        retry_sleep_seconds=args.agent_safetybench_judge_retry_sleep_seconds,
        fallback_model=args.agent_safetybench_judge_fallback_model,
        fallback_base_url=args.agent_safetybench_judge_fallback_base_url,
        fallback_api_key_env=args.agent_safetybench_judge_fallback_api_key_env,
        fallback_user_agent=args.agent_safetybench_judge_fallback_user_agent,
        fallback_request_retries=args.agent_safetybench_judge_fallback_request_retries,
        prompt_profile=args.agent_safetybench_judge_prompt_profile,
    )
    evolution_agent = (
        OpenAICompatibleEvolutionAgent(
            model=args.evolution_model,
            base_url=args.evolution_base_url,
            api_key_env=args.evolution_api_key_env,
            max_tokens=args.evolution_max_tokens,
            timeout_seconds=args.evolution_timeout_seconds,
            request_retries=args.evolution_request_retries,
            fallback_model=args.evolution_fallback_model,
            fallback_base_url=args.evolution_fallback_base_url,
            fallback_api_key=args.evolution_fallback_api_key,
            fallback_request_retries=args.evolution_fallback_request_retries,
            retry_sleep_seconds=args.evolution_retry_sleep_seconds,
            max_tool_rounds=args.evolution_max_tool_rounds,
            policy=selected_policy,
        )
        if args.evolution_mode == "model"
        else StaticEvolutionAgent(())
    )
    if memory_review.mode != "off":
        memory_review.reviewer = make_memory_reviewer(args)
    evolver = OnlineGraphEvolutionController(
        graph=graph,
        agent=evolution_agent,
        root=args.out_dir / "evolution",
        guard=guard,
        memory_store=state.memory_store,
        permission_store=state.permission_store,
        artifact_state=state.artifact_state,
        policy=selected_policy,
        skill_registry=skill_registry if args.skill_runtime else None,
        skill_evolution_interval=args.skill_evolution_interval,
        protect_existing_memory=memory_review.mode == "apply",
    )
    evolver.initialize()
    agent_security_bench_executor = LiveAgentSecurityBenchExecutor(
        dataset_dir=dataset,
        curated_path=curated_path,
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        target_request_retries=args.target_request_retries,
        target_retry_sleep_seconds=args.target_retry_sleep_seconds,
        target_fallback_model=args.target_fallback_model,
        target_fallback_base_url=args.target_fallback_base_url,
        target_fallback_api_key=args.target_fallback_api_key,
        target_fallback_request_retries=args.target_fallback_request_retries,
    )
    agent_safetybench_executor = LiveAgentSafetyBenchExecutor(
        dataset_dir=dataset,
        curated_path=curated_path,
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        target_request_retries=args.target_request_retries,
        target_retry_sleep_seconds=args.target_retry_sleep_seconds,
        target_fallback_model=args.target_fallback_model,
        target_fallback_base_url=args.target_fallback_base_url,
        target_fallback_api_key=args.target_fallback_api_key,
        target_fallback_request_retries=args.target_fallback_request_retries,
    )
    rows = existing_rows
    # Finish any committed boundary's pending summary BEFORE another Target task.
    skill_registry = _advance_consolidation_evidence_aware(consolidation_evidence_aware, args=args, rows=rows,
        state=state, evolver=evolver, registry=skill_registry)
    pending_batch = [
        _batch_episode_from_row(row=row, case=cases_by_id[str(row["task_id"])], graph=graph)
        for row in pending_batch_rows
    ]
    last_consolidation = (
        last_consolidation_from_ledger(evolver.root) if consolidation_evidence_aware is None and args.skill_consolidation_interval > 0 else {"ordinal": 0, "memory_pool_size": None}
    )

    for ordinal, case in enumerate(cases[len(rows):], start=len(rows) + 1):
        consolidation_due = (
            consolidation_evidence_aware is None
            and args.skill_consolidation_interval > 0
            and ordinal >= args.skill_consolidation_min_episodes
            and (
                ordinal % args.skill_consolidation_interval == 0
                or (
                    last_consolidation.get("memory_pool_size") is not None
                    and len(state.memory_store.items) - int(last_consolidation["memory_pool_size"])
                    >= args.skill_consolidation_min_new_lessons
                )
            )
        )
        # On consolidation episodes the per-episode Evolution Agent may not touch
        # skills; the dedicated consolidation round owns them after this episode.
        evolver.skill_updates_locked = consolidation_due

        if case.private.source_benchmark == "asb":
            executor = agent_security_bench_executor
            public_task = agent_security_bench_executor.public_task(case)
        elif case.private.source_benchmark == "agent_safetybench":
            executor = agent_safetybench_executor
            public_task = case.public
        else:
            raise ValueError(f"Unsupported source: {case.private.source_benchmark}")
        case_dir = args.out_dir / "raw_logs" / run_case_directory_name(
            ordinal=ordinal,
            task_id=case.public.task_id,
        )
        trace = JsonTraceStore(case_dir / "harness_trace.json")
        case_guard = _guard_for_case(guard, case_dir)
        harness = build_harness_from_graph(
            evolver.graph,
            guard=case_guard,
            trace_store=trace,
            memory_store=state.memory_store,
            skill_registry=skill_registry,
            permission_store=state.permission_store,
            prompt_addendum_template=state.artifact_state.prompt_addendum,
            guard_policy=state.artifact_state.guard_policy,
        )
        session = harness.start(public_task)
        selected_memory_items = tuple(session.state.retrieved_memory)
        session.state.annotations["run_ordinal"] = ordinal
        interaction_path: Path | None = None
        try:
            native = executor.run(case, session)
            if case.private.source_benchmark == "agent_safetybench":
                if args.agent_safetybench_full_judge:
                    judge_input = build_full_trajectory_judge_record(
                        case=case,
                        interaction=native.interaction,
                        harness_trace_records=trace.records,
                        native_row=native.row,
                        prompt_version=agent_safetybench_judge_config.prompt_version,
                    )
                    _write_json(case_dir / "agent_safetybench_judge_input.json", judge_input)
                    full_judge = judge_agent_safetybench_record(
                        record=judge_input,
                        config=agent_safetybench_judge_config,
                    )
                    native.row.update({
                        "agent_safetybench_main_judge": full_judge.get("judge"),
                        "agent_safetybench_main_judge_model": full_judge.get("judge_model"),
                        "agent_safetybench_judge_parse_ok": full_judge.get("parse_ok"),
                        "agent_safetybench_judge_label": full_judge.get("label"),
                        "agent_safetybench_judge_safe": full_judge.get("safe"),
                        "agent_safetybench_judge_unsafe": full_judge.get("unsafe"),
                        "agent_safetybench_judge_partial_unsafe": full_judge.get("partial_unsafe"),
                        "agent_safetybench_judge_full_unsafe": full_judge.get("full_unsafe"),
                        "agent_safetybench_task_completed": full_judge.get("task_completed"),
                        "agent_safetybench_completion_label": full_judge.get("completion_label"),
                        "agent_safetybench_refusal": full_judge.get("refusal"),
                        "agent_safetybench_judge_reasoning": full_judge.get("reasoning"),
                        "agent_safetybench_judge_unsafe_evidence": full_judge.get("unsafe_evidence"),
                        "agent_safetybench_judge_completion_evidence": full_judge.get("completion_evidence"),
                        "agent_safetybench_judge_error": full_judge.get("judge_error"),
                        "agent_safetybench_judge_endpoint": full_judge.get("judge_endpoint"),
                        "agent_safetybench_judge_base_url": full_judge.get("judge_base_url"),
                        "agent_safetybench_judge_attempts": full_judge.get("judge_attempts", []),
                    })
                    _write_json(case_dir / "agent_safetybench_judge_output.json", full_judge)
            outcome = source_outcome(case, native.row)
            feedback, delta = state.prepare_update(case=case, outcome=outcome, session=session)
            session.end(outcome)
            batch_episode = BatchEvolutionEpisode(
                case=case,
                session=session,
                feedback=feedback,
                trace_records=trace.records,
                next_version=delta.next_version,
            )
            pending_batch.append(batch_episode)
            should_evolve = args.evolution_batch_size == 1 or len(pending_batch) >= args.evolution_batch_size or ordinal == len(cases)
            if args.evolution_batch_size == 1:
                deployment = evolver.evolve(
                    case=case,
                    session=session,
                    feedback=feedback,
                    trace_records=trace.records,
                    next_version=delta.next_version,
                )
            elif should_evolve:
                deployment = evolver.evolve_batch(
                    episodes=pending_batch,
                    next_version=delta.next_version,
                )
                pending_batch = []
            else:
                deployment = _deferred_batch_deployment(
                    parent_version=evolver.graph.version,
                    next_version=delta.next_version,
                    task_id=case.public.task_id,
                    batch_size=args.evolution_batch_size,
                    pending_count=len(pending_batch),
                )
                evolver.graph = advance_graph_version(evolver.graph, version=delta.next_version)
                evolver._write_json(evolver.root / "versions" / delta.next_version / "graph.json", evolver.graph.to_dict())
            if evolver.artifact_state is not None:
                state.artifact_state = evolver.artifact_state
            if evolver.skill_registry is not None:
                skill_registry = evolver.skill_registry
                _persist_skill_registry(out_dir=args.out_dir, registry=skill_registry)
            state.commit_update(feedback, delta)
            interaction_path = case_dir / "interaction.json"
            _write_json(interaction_path, native.interaction)
        except Exception as exc:
            outcome = None
            feedback = None
            delta = None
            deployment = None
            failure = {"error_type": type(exc).__name__, "error": str(exc)}
            _write_json(case_dir / "failure.json", failure)
            row = {
                "ordinal": ordinal,
                "task_id": case.public.task_id,
                "source": case.private.source_benchmark,
                "status": "failed",
                "failure": failure,
                "harness_trace_path": str(trace.path),
                "harness_trace_verified": trace.verify(),
                "online_harness_version_before": state.version,
                "online_harness_version_after": state.version,
                "source_adapter": executor.source_adapter,
                "guard_delivery": "live_agent_visible_next_model_turn",
            }
            attach_memory_review(memory_review, row, selected=selected_memory_items,
                                 store=state.memory_store, rows=rows, case_dir=case_dir)
            if consolidation_evidence_aware is not None:
                row['skill_consolidation_evidence_aware'] = consolidation_evidence_aware.configuration
            metric_snapshot = _attach_latest_core_metrics(rows, row)
            rows.append(row)
            _append_jsonl(args.out_dir / "cases.jsonl", rows)
            _persist_core_metrics(args.out_dir, rows)
            _write_checkpoint(
                out_dir=args.out_dir,
                state=state,
                rows=rows,
                graph_hash=evolver.graph.graph_hash,
                skill_registry=skill_registry,
            )
            print(json.dumps({"event": "safecoevo_online_case_failed", "ordinal": ordinal, "total": len(cases), "task_id": case.public.task_id, **failure}, ensure_ascii=False), flush=True)
            print(json.dumps(_metrics_log_event(metric_snapshot), ensure_ascii=False), flush=True)
            skill_registry = _advance_consolidation_evidence_aware(consolidation_evidence_aware, args=args, rows=rows,
                state=state, evolver=evolver, registry=skill_registry)
            continue
        guard_counts = Counter(verdict.safety for verdict in session.state.guard_verdicts.values())
        row = {
            "ordinal": ordinal,
            "task_id": case.public.task_id,
            "source": case.private.source_benchmark,
            "status": outcome.status,
            "turn_count": native.row.get("turn_count"),
            "elapsed_seconds": native.row.get("elapsed_seconds"),
            "official_feedback": feedback.to_dict(),
            "unified_metrics": _unified_metrics_from_feedback(feedback),
            "native_evaluator_result": _safe_native_result_fields(case.private.source_benchmark, native.row),
            **_top_level_audit_fields(case.private.source_benchmark, native.row),
            "harness_delta": delta.to_dict(),
            "processor_graph_deployment": deployment.to_dict(),
            "raw_trace_path": str(interaction_path),
            "harness_trace_path": str(trace.path),
            "harness_trace_verified": trace.verify(),
            "guard_summary": {
                "events": len(session.state.guard_verdicts),
                "safety": dict(sorted(guard_counts.items())),
                "available": sum(verdict.availability == "available" for verdict in session.state.guard_verdicts.values()),
            },
            "retrieved_memory_ids": [item.memory_id for item in session.state.retrieved_memory],
            "memory_retrieval": session.state.annotations.get("memory_retrieval", {}),
            "skill_runtime": session.state.annotations.get("skill_runtime_snapshot", {}),
            "skills_advertised": (session.state.annotations.get("skill_runtime") or {}).get("skills_advertised", []),
            "skill_retrieval": session.state.annotations.get("skill_retrieval", {}),
            "skill_usage": session.state.annotations.get("skill_usage", []),
            "guard_exemptions": session.state.annotations.get("guard_exemptions", []),
            "committed_memory_ids": session.state.annotations.get("committed_memory_ids", []),
            "online_harness_version_before": feedback.harness_version_before,
            "online_harness_version_after": state.version,
            "processor_graph_hash_after": evolver.graph.graph_hash,
            "source_adapter": executor.source_adapter,
            "guard_delivery": "live_agent_visible_next_model_turn",
        }
        evolver.finalize_episode_storage(deployment=deployment, ordinal=ordinal)
        row["processor_graph_deployment"] = deployment.to_dict()
        attach_memory_review(memory_review, row, selected=selected_memory_items,
                             store=state.memory_store, rows=rows, case_dir=case_dir,
                             case=case, session=session, feedback=feedback, trace_records=trace.records)
        if consolidation_evidence_aware is not None:
            row['skill_consolidation_evidence_aware'] = consolidation_evidence_aware.configuration
        metric_snapshot = _attach_latest_core_metrics(rows, row)
        rows.append(row)
        evolution_rows.append(
            {
                "feedback": feedback.to_dict(),
                "memory_delta": delta.to_dict(),
                "processor_graph_deployment": deployment.to_dict(),
            }
        )
        _append_jsonl(args.out_dir / "cases.jsonl", rows)
        _append_jsonl(args.out_dir / "evolution_log.jsonl", evolution_rows)
        _persist_core_metrics(args.out_dir, rows)
        _write_checkpoint(
            out_dir=args.out_dir,
            state=state,
            rows=rows,
            graph_hash=evolver.graph.graph_hash,
            skill_registry=skill_registry,
        )
        print(json.dumps({"event": "safecoevo_online_case_end", "ordinal": ordinal, "total": len(cases), "task_id": case.public.task_id, "source": case.private.source_benchmark, "status": outcome.status, "harness_version_after": state.version, "guard_events": len(session.state.guard_verdicts), "safety": feedback.safety, "goal": feedback.goal, "memory_delta": delta.change_type, "graph_deployment": deployment.status}, ensure_ascii=False), flush=True)
        print(json.dumps(_metrics_log_event(metric_snapshot), ensure_ascii=False), flush=True)
        skill_registry = _advance_consolidation_evidence_aware(consolidation_evidence_aware, args=args, rows=rows,
            state=state, evolver=evolver, registry=skill_registry)
        if consolidation_due:
            memory_items = [
                {
                    "memory_id": item.memory_id,
                    "content": item.content,
                    "tags": list(item.tags),
                    "evidence_trace_hashes": list(item.evidence_trace_hashes),
                    "verified": bool(item.verified),
                }
                for item in state.memory_store.items
            ]
            consolidation_result = run_consolidation_round(
                agent=evolver.agent,
                registry=skill_registry,
                memory_items=memory_items,
                rows=rows,
                ordinal=ordinal,
                root=evolver.root,
                since_ordinal=int(last_consolidation.get("ordinal") or 0),
            )
            evolver.skill_updates_locked = False
            last_consolidation = {"ordinal": ordinal, "memory_pool_size": len(memory_items)}
            if isinstance(consolidation_result.get("registry"), SkillRegistry):
                skill_registry = consolidation_result["registry"]
                evolver.skill_registry = skill_registry
                _persist_skill_registry(out_dir=args.out_dir, registry=skill_registry)
                _write_checkpoint(
                    out_dir=args.out_dir,
                    state=state,
                    rows=rows,
                    graph_hash=evolver.graph.graph_hash,
                    skill_registry=skill_registry,
                )
            print(json.dumps({"event": "skill_consolidation", **{k: v for k, v in consolidation_result.items() if k != "registry"}}, ensure_ascii=False, default=str), flush=True)
    completed_feedback = [row["official_feedback"] for row in rows if isinstance(row.get("official_feedback"), dict)]
    final_core_metrics = _persist_core_metrics(args.out_dir, rows)
    _write_json(
        args.out_dir / "summary.json",
        {
            "protocol": plan["protocol"],
            "completed_cases": len(rows),
            "final_harness_version": state.version,
            "final_processor_graph_hash": evolver.graph.graph_hash,
            "verified_memories": len(state.memory_store.items),
            "active_skill_registry_version": skill_registry.version,
            "active_skill_registry_hash": skill_registry.registry_hash,
            "active_skills": len(skill_registry.active),
            "processor_graph_deployments": dict(sorted(Counter(str(row.get("processor_graph_deployment", {}).get("status")) for row in rows if isinstance(row.get("processor_graph_deployment"), dict)).items())),
            "status": dict(sorted(Counter(str(row.get("status")) for row in rows).items())),
            "feedback_safety": dict(sorted(Counter(str(row["safety"]) for row in completed_feedback).items())),
            "feedback_goal": dict(sorted(Counter(str(row["goal"]) for row in completed_feedback).items())),
            "core_metrics": final_core_metrics,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
