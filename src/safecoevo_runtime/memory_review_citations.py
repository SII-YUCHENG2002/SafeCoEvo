"""Bounded, source-aware citation repair diagnostics for memory review.

This module does not deliver evidence, reinterpret citations, or alter decisions.
It distinguishes visible metadata from observations the reviewer may cite.
"""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .memory_review_evidence import EvidencePack


_CITABLE_ROOTS = {"task", "public_trace", "official_feedback"}
_ROOT_PRIORITY = {"official_feedback": 0, "public_trace": 1, "task": 2}
_MAX_ISSUES = 32
_STRING_LIMITS = {"evidence_id": 128, "memory_id": 128, "root": 128, "pointer": 512}


def _citable(entry: dict[str, Any]) -> bool:
    value = entry.get("value")
    if entry.get("root") not in _CITABLE_ROOTS or value is None:
        return False
    return bool(value.strip()) if isinstance(value, str) else True


def _bounded_fields(fields: dict[str, str], limits: dict[str, int]) -> dict[str, Any]:
    result: dict[str, Any] = dict(fields)
    truncated = []
    lengths = {}
    for name, limit in limits.items():
        value = result.get(name)
        if isinstance(value, str) and len(value) > limit:
            result[name] = value[:limit - 3] + "..."
            truncated.append(name)
            lengths[name] = len(value)
    if truncated:
        result["truncated_fields"] = truncated
        result["original_lengths"] = lengths
    return result


def _reference_fields(decision: dict[str, Any]):
    refs = decision.get("evidence_ids")
    if isinstance(refs, list):
        for index, value in enumerate(refs):
            yield f"evidence_ids[{index}]", value
    claims = decision.get("revoked_claims")
    if isinstance(claims, list):
        for claim_index, claim in enumerate(claims):
            if not isinstance(claim, dict) or not isinstance(claim.get("evidence_ids"), list):
                continue
            for index, value in enumerate(claim["evidence_ids"]):
                yield f"revoked_claims[{claim_index}].evidence_ids[{index}]", value


def citation_issues(decision: Any, pack: EvidencePack, delivered_ids: set[str]) -> list[dict]:
    """Report all invalid string references, bounded without silently accepting any.

    Malformed structures remain the original validator's responsibility. A
    summary replaces the last detail if there are more than 32 issues, so a
    nonempty result always blocks a decision with unresolved citations.
    """
    if not isinstance(decision, dict):
        return []
    memory_id = decision.get("memory_id")
    memory_id = memory_id if isinstance(memory_id, str) else ""
    issues: list[dict] = []
    total = 0
    for field, evidence_id in _reference_fields(decision):
        if not isinstance(evidence_id, str):
            continue
        entry = pack.entries.get(evidence_id)
        if entry is None:
            code = "unknown_evidence_id"
            message = "This evidence ID does not exist in this episode's evidence archive."
            repair = ("Recheck the citation against this episode's citation directory. "
                      "Use an exact supported ID; no automatic substitution is permitted.")
        elif not _citable(entry):
            code = "non_citable_evidence_id"
            message = "This ID exists but is metadata or an empty value, not a citable observation."
            repair = ("Reading this ID cannot make it citable. Reassess the claim and cite "
                      "supporting task, public_trace, or official_feedback observations instead.")
        elif evidence_id not in delivered_ids:
            code = "undelivered_evidence_id"
            message = "This ID names a citable observation whose body has not been delivered."
            repair = ("Request this exact ID with read and inspect the returned evidence "
                      "before citing it, or reassess using already delivered evidence.")
        else:
            continue
        total += 1
        if len(issues) >= _MAX_ISSUES:
            continue
        item = {"code": code, "field": field, "message": message,
                "memory_id": memory_id, "evidence_id": evidence_id, "repair": repair}
        if entry is not None:
            item.update(root=entry["root"], pointer=entry["pointer"])
        issues.append(_bounded_fields(item, _STRING_LIMITS))
    if total > _MAX_ISSUES:
        issues[-1] = _bounded_fields({
            "code": "citation_issue_limit_exceeded", "field": "evidence_ids",
            "message": "Additional invalid citations were omitted from this bounded diagnostic.",
            "memory_id": memory_id, "total_errors": total,
            "omitted_errors": total - (_MAX_ISSUES - 1),
            "repair": "Reassess all citation fields; unresolved omitted issues still reject this decision.",
        }, _STRING_LIMITS)
    return issues


def citation_context(pack: EvidencePack, delivered_ids: set[str], max_chars: int = 6000) -> dict:
    """Provide bounded source metadata for already delivered citable observations.

    This directory is not a delivery and contains no evidence body. Its ordering
    puts official feedback before trajectory and task sources. Truncated paths
    are labeled, and ``omitted`` reports all entries not included in the budget.
    """
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("Citation context max_chars must be a positive integer")
    entries = [(eid, entry) for eid, entry in pack.entries.items()
               if eid in delivered_ids and _citable(entry)]
    entries.sort(key=lambda item: (_ROOT_PRIORITY[item[1]["root"]], item[0]))
    result = {
        "delivery": "citation_directory_only",
        "semantic_correctness_proven": False,
        "notice": "Source metadata only; a valid ID does not prove a claim or causal effect.",
        "entries": [], "total": len(entries), "omitted": len(entries),
    }

    def size(value: dict) -> int:
        return len(json.dumps(value, ensure_ascii=False, allow_nan=False))

    if size(result) > max_chars:
        raise ValueError("Citation context metadata cannot fit max_chars")
    for eid, entry in entries:
        item = _bounded_fields({"id": eid, "root": entry["root"], "pointer": entry["pointer"]},
                               {"root": 128, "pointer": 512})
        candidate = {**result, "entries": [*result["entries"], item],
                     "omitted": result["omitted"] - 1}
        if size(candidate) > max_chars:
            continue
        result = candidate
    return result


def fit_repair_errors(errors: list[dict], *, max_chars: int) -> dict:
    """Bound transmitted diagnostics, preserving each memory's first error.

    Full validation errors stay in the caller's audit record. This only creates
    an explicitly incomplete transmission copy; omitted diagnostics never grant
    validation success. Re-fitting its output retains the original error count.
    """
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("Repair error max_chars must be a positive integer")
    if not isinstance(errors, list) or any(not isinstance(error, dict) for error in errors):
        raise ValueError("Repair errors must be a list of objects")
    originals, inherited_omissions = [], 0
    for error in errors:
        if (error.get("code") == "repair_errors_omitted"
                and type(error.get("omitted_errors")) is int
                and error["omitted_errors"] > 0):
            inherited_omissions += error["omitted_errors"]
        else:
            originals.append(error)
    total = len(originals) + inherited_omissions

    def result_for(indices):
        chosen = [deepcopy(originals[index]) for index in sorted(indices)]
        omitted = total - len(chosen)
        if omitted:
            chosen.append({
                "code": "repair_errors_omitted", "field": "review",
                "message": "Additional validation errors were omitted from this bounded feedback. "
                           "Recheck all citation and review fields; omitted errors still reject a decision.",
                "total_errors": total, "omitted_errors": omitted,
            })
        return {"errors": chosen, "errors_truncated": bool(omitted),
                "errors_total": total, "errors_omitted": omitted}

    def fits(result):
        return len(json.dumps(result, ensure_ascii=False, allow_nan=False)) <= max_chars

    full = result_for(range(len(originals)))
    if fits(full):
        return full
    first_indices, seen_memories = set(), set()
    for index, error in enumerate(originals):
        memory_id = error.get("memory_id")
        memory_id = memory_id if isinstance(memory_id, str) else None
        if memory_id not in seen_memories:
            first_indices.add(index)
            seen_memories.add(memory_id)
    bounded = result_for(first_indices)
    if not fits(bounded):
        raise ValueError("First validation error for each memory cannot fit repair budget")
    selected = first_indices
    for index in range(len(originals)):
        if index in selected:
            continue
        candidate = result_for(selected | {index})
        if fits(candidate):
            selected.add(index)
            bounded = candidate
    return bounded
