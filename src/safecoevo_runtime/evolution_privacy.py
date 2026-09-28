"""Source-blind views used exclusively by the Evolution Agent.

The coordinator needs benchmark provenance to select an executor and invoke an
official oracle.  An Evolution Agent does not: it should reason from a task's
actual public interaction, the normalized outcome, and the active Harness
mechanisms.  This module builds that narrower view without changing the
coordinator's private records or the target Agent's task payload.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from .evolution_graph import ProcessorGraph, ProcessorNode
from .online_feedback import OfficialEpisodeFeedback


SOURCE_BLIND_VIEW_VERSION = 1

# These are provenance labels, not task semantics. The complete task trace is
# retained; only an explicit benchmark identifier is redacted if it appears in
# extensible runtime metadata or natural-language content.
_SOURCE_IDENTITY_PATTERN = re.compile(
    r"agent[ _-]?safetybench|agent[ _-]?dojo|agent[ _-]?dyn|agent[ _-]?harm|\basb\b|(?<![A-Za-z0-9])asb(?=$|_|(?-i:[A-Z]))",
    flags=re.IGNORECASE,
)


def contains_source_identity(value: Any) -> bool:
    """Return whether metadata/code names a known source benchmark."""

    if isinstance(value, str):
        return bool(_SOURCE_IDENTITY_PATTERN.search(value))
    if isinstance(value, dict):
        return any(
            contains_source_identity(str(key)) or contains_source_identity(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_source_identity(item) for item in value)
    return False


def scrub_source_identity_text(value: str) -> str:
    """Replace only benchmark provenance words with a neutral identifier."""

    return _SOURCE_IDENTITY_PATTERN.sub("benchmark", value)


def scrub_source_identity(value: Any) -> Any:
    """Deep-copy JSON-like data while removing benchmark provenance labels."""

    if isinstance(value, str):
        return scrub_source_identity_text(value)
    if isinstance(value, dict):
        return {
            scrub_source_identity_text(str(key)): scrub_source_identity(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [scrub_source_identity(item) for item in value]
    if isinstance(value, tuple):
        return [scrub_source_identity(item) for item in value]
    return copy.deepcopy(value)


def _replace_processor_ids(value: Any, aliases: dict[str, str]) -> Any:
    """Replace private graph IDs wherever they occur in trace metadata."""

    if isinstance(value, str):
        # Longest-first prevents a shorter processor ID from partially
        # replacing a longer one with the same prefix.
        for actual, alias in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
            value = value.replace(actual, alias)
        return value
    if isinstance(value, dict):
        return {
            _replace_processor_ids(str(key), aliases): _replace_processor_ids(item, aliases)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_processor_ids(item, aliases) for item in value]
    if isinstance(value, tuple):
        return [_replace_processor_ids(item, aliases) for item in value]
    return copy.deepcopy(value)


def source_blind_feedback(feedback: OfficialEpisodeFeedback) -> dict[str, Any]:
    """Expose one normalized official outcome, never its native oracle schema."""

    return {
        "schema_version": SOURCE_BLIND_VIEW_VERSION,
        "task_id": feedback.task_id,
        "harness_version_before": feedback.harness_version_before,
        "evaluation_complete": feedback.evaluation_complete,
        "outcomes": {
            "safety": feedback.safety,
            "goal": feedback.goal,
        },
        "availability": {
            "safety": bool(feedback.availability.get("safety", False)),
            "goal": bool(feedback.availability.get("goal", False)),
        },
        "trace_summary": {
            "turn_count": feedback.trace_summary.get("turn_count"),
            "guard_observations": copy.deepcopy(feedback.trace_summary.get("guard_observations", {})),
        },
        "feedback_contract": "normalized_official_outcome_without_benchmark_or_oracle_identity",
    }


def source_blind_trace(
    trace_records: list[dict[str, Any]],
    *,
    processor_id_aliases: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Copy a complete trace while removing evaluator-only provenance metadata.

    The original append-only trace remains private coordinator evidence.  This
    derived view retains every task message, tool definition/result, action,
    Guard verdict, and Processor effect, but replaces final native outcome
    metadata with source-neutral outcome fields.  Its hash chain is rebound so
    Evolution can still audit the exact derived records it received.
    """

    aliases = processor_id_aliases or {}
    result: list[dict[str, Any]] = []
    previous_hash = ""
    for raw in trace_records:
        # Processor observations and annotations are extensible. Apply the
        # same aliasing/de-identification recursively rather than relying on
        # a fixed list of trace fields to remain source-blind over time.
        record = scrub_source_identity(_replace_processor_ids(raw, aliases))
        record.pop("entry_hash", None)
        record.pop("previous_hash", None)
        outcome = record.get("outcome")
        if isinstance(outcome, dict):
            record["outcome"] = {
                "status": outcome.get("status"),
                "unsafe_outcome": outcome.get("unsafe_outcome"),
                "legitimate_task_success": outcome.get("legitimate_task_success"),
                "verified": outcome.get("verified"),
            }
        record["previous_hash"] = previous_hash
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        entry_hash = hashlib.sha256(encoded).hexdigest()
        record["entry_hash"] = entry_hash
        previous_hash = entry_hash
        result.append(record)
    return result


@dataclass(frozen=True)
class SourceBlindGraphView:
    """A graph view plus the reversible IDs needed by the private controller."""

    graph: ProcessorGraph
    view_to_actual_processor_id: dict[str, str]

    @property
    def actual_to_view_processor_id(self) -> dict[str, str]:
        return {actual: view for view, actual in self.view_to_actual_processor_id.items()}

    def translate_proposal_to_actual(self, proposal: dict[str, Any]) -> dict[str, Any]:
        """Map references to anonymized active nodes back at deployment time."""

        translated = copy.deepcopy(proposal)
        edits = translated.get("edits")
        if not isinstance(edits, list):
            return translated
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            processor_id = edit.get("processor_id")
            if isinstance(processor_id, str):
                edit["processor_id"] = self.view_to_actual_processor_id.get(processor_id, processor_id)
            after = edit.get("after")
            if isinstance(after, list):
                edit["after"] = [
                    self.view_to_actual_processor_id.get(item, item) if isinstance(item, str) else item
                    for item in after
                ]
        return translated


def source_blind_graph_view(graph: ProcessorGraph) -> SourceBlindGraphView:
    """Hide benchmark-bearing active node names while retaining full behavior.

    Only nodes whose identifiers, class names, or source text name a benchmark
    receive aliases.  Generic built-ins and generic generated Processor names
    stay readable, so the Evolution Agent can still make precise graph edits.
    """

    aliases: dict[str, str] = {}
    reserved = {node.processor_id for node in graph.nodes}
    counter = 1
    for node in graph.nodes:
        fields = (node.processor_id, node.processor_name, node.exported_class or "", node.source or "")
        if not any(contains_source_identity(value) for value in fields):
            continue
        while True:
            alias = f"source_blind_processor_{counter:03d}"
            counter += 1
            if alias not in reserved:
                break
        aliases[node.processor_id] = alias
        reserved.add(alias)

    actual_to_view = dict(aliases)
    nodes: list[ProcessorNode] = []
    for node in graph.nodes:
        view_id = actual_to_view.get(node.processor_id, node.processor_id)
        source = node.source
        exported_class = node.exported_class
        processor_name = node.processor_name
        if node.processor_id in aliases:
            ordinal = list(aliases).index(node.processor_id) + 1
            processor_name = f"GeneratedProcessor{ordinal:03d}"
            if exported_class is not None:
                replacement_class = f"GeneratedProcessor{ordinal:03d}"
                if source is not None:
                    source = source.replace(exported_class, replacement_class)
                exported_class = replacement_class
        if source is not None:
            for actual, view in sorted(actual_to_view.items(), key=lambda item: len(item[0]), reverse=True):
                source = source.replace(actual, view)
            source = scrub_source_identity_text(source)
        nodes.append(
            ProcessorNode(
                processor_id=view_id,
                processor_name=scrub_source_identity_text(processor_name),
                implementation=node.implementation,
                hooks=node.hooks,
                order=node.order,
                parameters=scrub_source_identity(node.parameters),
                source=source,
                exported_class=scrub_source_identity_text(exported_class) if exported_class else None,
                after=tuple(actual_to_view.get(item, item) for item in node.after),
            )
        )
    return SourceBlindGraphView(
        graph=ProcessorGraph(version=graph.version, nodes=tuple(nodes), schema_version=graph.schema_version),
        view_to_actual_processor_id={view: actual for actual, view in aliases.items()},
    )
