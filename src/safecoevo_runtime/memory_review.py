"""Shared evidence validation and guarded memory revision.

Citation validation proves provenance, NOT semantic entailment or causality.
Semantic judgments remain explicitly attributed to the reviewer model.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import PurePosixPath
from typing import Any

from .contracts import MemoryItem
from .evolution_privacy import contains_source_identity
from .memory import memory_identity

EDIT_ACTIONS = {"refine_scope", "correct_error"}
EFFECTS = {"helpful", "harmful", "neutral", "unknown"}


def memory_snapshot(item: MemoryItem) -> dict[str, Any]:
    return {**asdict(item), **memory_identity(item)}


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _quote(value: Any, container: str, name: str) -> str:
    value = _text(value, name)
    if value not in container:
        raise ValueError(f"{name} is not an exact source quote")
    return value


def _resolve_pointer(evidence: dict[str, Any], pointer: str) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("Evidence pointer must be an absolute JSON pointer")
    value: Any = evidence
    try:
        for token in pointer[1:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                if not token.isdecimal():
                    raise ValueError("Invalid array index")
                value = value[int(token)]
            else:
                value = value[token]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("Evidence pointer does not resolve") from exc
    return value


def _validate_evidence(refs: Any, evidence: dict[str, Any], *, required: bool) -> None:
    if not isinstance(refs, list):
        raise ValueError("evidence must be a list")
    roots = set()
    for ref in refs:
        if not isinstance(ref, dict):
            raise ValueError("Each evidence reference must be an object")
        pointer = ref.get("pointer")
        value = _resolve_pointer(evidence, pointer)
        root = pointer.split("/")[1]
        if root not in {"task", "public_trace", "official_feedback"}:
            raise ValueError("Cite task, current trajectory or released feedback, not reviewer opinions")
        # String leaves only: citing an entire object/array could hide which
        # observation supports the claim. Booleans may be cited canonically.
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif isinstance(value, (int, float)):
            value = str(value)
        if not isinstance(value, str):
            raise ValueError("Evidence references must point to scalar leaves")
        _quote(ref.get("quote"), value, "evidence quote")
        roots.add(root)
    if required and not {"public_trace", "official_feedback"} <= roots:
        raise ValueError("Revision requires current trajectory AND official feedback references")


def validate_normalized_decision(decision: dict[str, Any], evidence: dict[str, Any],
                                 old: MemoryItem) -> dict[str, Any]:
    """Validate one evidence-resolved decision before any live memory mutation."""
    if not isinstance(decision, dict) or decision.get("memory_id") != old.memory_id:
        raise ValueError("Review must refer to the selected memory")
    if decision.get("fingerprint") != memory_identity(old)["fingerprint"]:
        raise ValueError("Review memory version differs from injected version")
    action = decision.get("action")
    if action not in EDIT_ACTIONS | {"keep", "uncertain"}:
        raise ValueError("Unknown review action")
    for field in ("safety_effect", "goal_effect"):
        if decision.get(field) not in EFFECTS:
            raise ValueError(f"Invalid {field}")
    _text(decision.get("reason"), "reason")
    _text(decision.get("alternative_explanations"), "alternative_explanations")
    _validate_evidence(decision.get("evidence"), evidence, required=action in EDIT_ACTIONS)
    if action not in EDIT_ACTIONS:
        if decision.get("new_content") not in (None, ""):
            raise ValueError("Keep/uncertain cannot modify memory")
        return deepcopy(decision)
    _quote(decision.get("problem_claim"), old.content, "problem_claim")
    new = _text(decision.get("new_content"), "new_content")
    if new == old.content or len(new) > 12000:
        raise ValueError("Revision must change content and remain compact (<=12000 characters)")
    if contains_source_identity(new):
        raise ValueError("Revised memory must be source-blind")
    preserved = decision.get("preserved_claims")
    revoked = decision.get("revoked_claims")
    if not isinstance(preserved, list) or not isinstance(revoked, list):
        raise ValueError("preserved_claims/revoked_claims must be lists")
    for claim in preserved:
        if not isinstance(claim, dict):
            raise ValueError("Preserved claim must be an object")
        _quote(claim.get("old_quote"), old.content, "preserved old claim")
        _quote(claim.get("new_quote"), new, "preserved new claim")
    if action == "refine_scope":
        if not preserved or revoked or decision.get("whole_memory_wrong") is not False:
            raise ValueError("Scope refinement must preserve claims and cannot revoke semantics")
    else:
        if not revoked:
            raise ValueError("Error correction needs explicit contradicted claims")
        for claim in revoked:
            if not isinstance(claim, dict):
                raise ValueError("Revoked claim must be an object")
            _quote(claim.get("quote"), old.content, "revoked claim")
            _text(claim.get("reason"), "revocation reason")
            _validate_evidence(claim.get("evidence"), evidence, required=True)
        if decision.get("whole_memory_wrong") is True:
            if not any(claim["quote"] == old.content for claim in revoked):
                raise ValueError("Whole-memory replacement must explicitly address the entire old content")
        elif decision.get("whole_memory_wrong") is not False or not preserved:
            raise ValueError("Partial correction must preserve remaining valid claims")
    return deepcopy(decision)


def apply_revisions(store: Any, selected: list[MemoryItem] | tuple[MemoryItem, ...],
                    decisions: list[dict[str, Any]], trace_hashes: tuple[str, ...]) -> list[dict[str, Any]]:
    """Apply already validated decisions, all-or-nothing, for future episodes.

    No deletion, tag rewriting, verified-status escalation, or utility transfer.
    The runner persists these before/after records with its checkpoint commit.
    """
    old_by_id = {item.memory_id: item for item in selected}
    live = list(store.items)
    indices = {item.memory_id: index for index, item in enumerate(live)}
    if len(indices) != len(live):
        raise ValueError("Duplicate live memory IDs")
    changes = []
    for decision in decisions:
        if decision["action"] not in EDIT_ACTIONS:
            continue
        old = old_by_id[decision["memory_id"]]
        index = indices.get(old.memory_id)
        if index is None or memory_identity(live[index]) != memory_identity(old):
            raise ValueError("Live memory version changed since injection")
        current = live[index]
        new = replace(current, content=decision["new_content"],
                      evidence_trace_hashes=tuple(dict.fromkeys((*current.evidence_trace_hashes, *trace_hashes))))
        live[index] = new
        changes.append({"action": decision["action"], "before": memory_snapshot(current), "after": memory_snapshot(new)})
    # This is the only mutation, after every candidate's live-version check.
    store.items[:] = live
    return changes


def validate_memory_write_boundary(patch: dict[str, Any], store: Any) -> None:
    """Reserve existing-memory revisions for the independent reviewer in apply mode."""
    existing = {item.memory_id for item in store.items}
    for update in patch.get("file_updates", []):
        path = str(update.get("path", "")).strip().replace("\\", "/")
        if path.startswith("artifacts/"):
            path = path[len("artifacts/"):]
        if str(PurePosixPath(path)) != "memory/validated_experience.jsonl":
            continue
        if update.get("mode") != "append_jsonl":
            raise ValueError("Existing memory revisions are reserved for independent memory review")
        for record in update.get("records", []):
            if record.get("memory_id") in existing:
                raise ValueError("Existing memory ID is protected by independent memory review")
