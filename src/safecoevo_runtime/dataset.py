"""Private and public dataset adapter for SafeCoEvo streams."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .events import PublicTask

@dataclass(frozen=True)
class PrivateEvaluator:
    """Coordinator/executor-only fields; never pass this object to a target agent."""

    task_id: str
    source_benchmark: str
    source_case_key: str
    source_adapter: str
    record: dict[str, Any]


@dataclass(frozen=True)
class SafeCoEvoCase:
    """A joined public task and private evaluator record with an explicit boundary."""

    public: PublicTask
    private: PrivateEvaluator


def _read_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"Expected a JSON array of objects: {path}")
    return value


def _agent_safetybench_judge_metadata(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    """Load official task metadata for the Agent-SafetyBench trajectory judge.

    This data remains on the evaluator side of the public/private boundary.  It
    supplies task semantics (not trajectory outcomes) and is never exposed to
    the target agent.
    """

    path = dataset_dir / "records.jsonl"
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        source = row.get("source") if isinstance(row, dict) else None
        source_key = source.get("source_case_key") if isinstance(source, dict) else None
        labels = row.get("labels") if isinstance(row, dict) else None
        source_labels = labels.get("source_labels") if isinstance(labels, dict) else None
        scenario = row.get("scenario") if isinstance(row, dict) else None
        attack = scenario.get("attack") if isinstance(scenario, dict) else None
        if not isinstance(source_key, str) or not isinstance(source_labels, dict):
            continue
        risks = source_labels.get("risks")
        fulfillable = source_labels.get("fulfillable")
        if isinstance(fulfillable, int) and fulfillable in {0, 1}:
            fulfillable = bool(fulfillable)
        # The normalizer's canonical attack channel is an evaluator-side
        # analysis label.  Do not reuse it in a Judge metadata record: only the
        # released task's own attack declaration may describe this context.
        condition = "direct_user_request" if isinstance(attack, dict) and attack.get("present") is False else "unknown"
        result[source_key] = {
            "source": "official_records_jsonl",
            "risks": [str(item) for item in risks] if isinstance(risks, list) else [],
            "failure_modes": source_labels.get("failure_modes"),
            "fulfillable": fulfillable,
            "attack_context": {
                "condition": condition,
                "attack_present": bool(attack.get("present")) if isinstance(attack, dict) else False,
            },
        }
    return result


def _require_string(row: dict[str, Any], field: str, *, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} requires non-empty string {field!r}")
    return value


def _public_task(row: dict[str, Any]) -> PublicTask:
    forbidden = {"provenance", "evaluation", "attack_schedule", "analysis_labels", "tool_binding", "source"}
    leaked = forbidden.intersection(row)
    if leaked:
        raise ValueError(f"Agent view leaks evaluator-only fields: {sorted(leaked)}")
    task_id = _require_string(row, "task_id", label="agent view")
    system = row.get("system")
    messages = row.get("messages")
    tools = row.get("tools")
    runtime = row.get("runtime")
    if not isinstance(system, dict) or system.get("role") != "system" or not isinstance(system.get("content"), str):
        raise ValueError(f"{task_id}: invalid system prompt")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{task_id}: messages must be a non-empty list")
    if not isinstance(tools, list):
        raise ValueError(f"{task_id}: tools must be a list")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("max_turns"), int):
        raise ValueError(f"{task_id}: runtime.max_turns must be an integer")
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(name, str) or not name:
            raise ValueError(f"{task_id}: each tool needs function.name")
        if name in names:
            raise ValueError(f"{task_id}: duplicate public tool name {name!r}")
        names.add(name)
    return PublicTask(
        task_id=task_id,
        system=copy.deepcopy(system),
        messages=tuple(copy.deepcopy(messages)),
        tools=tuple(copy.deepcopy(tools)),
        runtime=copy.deepcopy(runtime),
    )


def _private_evaluator(row: dict[str, Any], *, public_tool_names: set[str]) -> PrivateEvaluator:
    task_id = _require_string(row, "task_id", label="evaluator view")
    provenance = row.get("provenance")
    evaluation = row.get("evaluation")
    binding = row.get("tool_binding")
    if not isinstance(provenance, dict) or not isinstance(evaluation, dict) or not isinstance(binding, dict):
        raise ValueError(f"{task_id}: malformed evaluator view")
    source = _require_string(provenance, "source_benchmark", label=f"{task_id}.provenance")
    source_case_key = _require_string(provenance, "source_case_key", label=f"{task_id}.provenance")
    adapter = _require_string(evaluation, "source_adapter", label=f"{task_id}.evaluation")
    if set(binding) != public_tool_names:
        raise ValueError(f"{task_id}: private tool binding does not match public tools")
    return PrivateEvaluator(
        task_id=task_id,
        source_benchmark=source,
        source_case_key=source_case_key,
        source_adapter=adapter,
        record=copy.deepcopy(row),
    )


def load_safecoevo_dataset(dataset_dir: Path) -> list[SafeCoEvoCase]:
    """Load a stream while preserving the agent-view/evaluator-view isolation."""

    dataset_dir = dataset_dir.resolve()
    agents = _read_array(dataset_dir / "agent_view.json")
    evaluators = _read_array(dataset_dir / "evaluator_view.json")
    judge_metadata_by_source = _agent_safetybench_judge_metadata(dataset_dir)
    evaluator_by_id: dict[str, dict[str, Any]] = {}
    for row in evaluators:
        task_id = _require_string(row, "task_id", label="evaluator view")
        if task_id in evaluator_by_id:
            raise ValueError(f"Duplicate evaluator task_id: {task_id}")
        evaluator_by_id[task_id] = row

    cases: list[SafeCoEvoCase] = []
    public_ids: set[str] = set()
    for row in agents:
        public = _public_task(row)
        if public.task_id in public_ids:
            raise ValueError(f"Duplicate agent task_id: {public.task_id}")
        public_ids.add(public.task_id)
        evaluator = evaluator_by_id.get(public.task_id)
        if evaluator is None:
            raise ValueError(f"Missing evaluator record for {public.task_id}")
        names = {str(tool["function"]["name"]) for tool in public.tools}
        evaluator_for_private = copy.deepcopy(evaluator)
        provenance = evaluator_for_private.get("provenance")
        source_key = provenance.get("source_case_key") if isinstance(provenance, dict) else None
        metadata = judge_metadata_by_source.get(source_key) if isinstance(source_key, str) else None
        if metadata is not None:
            evaluator_for_private["judge_task_metadata"] = metadata
        cases.append(SafeCoEvoCase(public=public, private=_private_evaluator(evaluator_for_private, public_tool_names=names)))
    extra = set(evaluator_by_id) - public_ids
    if extra:
        raise ValueError(f"Evaluator-only task IDs are not executable: {sorted(extra)[:5]}")
    return cases
