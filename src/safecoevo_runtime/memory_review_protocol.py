"""Evidence-ID review decisions with shared memory safety validation.

An evidence ID is usable only after this reviewer received it. Resolving that ID
and checking its archived scalar leaf proves provenance, not semantic entailment
or a memory's causal contribution to an outcome.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from .contracts import MemoryItem
from .memory_review import EDIT_ACTIONS, EFFECTS, _validate_evidence, validate_normalized_decision
from .memory import memory_identity


class ReviewValidationError(ValueError):
    """Small, safe repair feedback, without provider requests or raw responses."""

    def __init__(self, code: str, field: str, message: str, memory_id: str):
        self.details = {
            "code": code, "field": field, "message": message, "memory_id": memory_id,
        }
        super().__init__(message)


_FIELDS = {
    "memory_id", "fingerprint", "action", "new_content", "safety_effect", "goal_effect",
    "reason", "alternative_explanations", "evidence_ids", "problem_claim",
    "preserved_claims", "revoked_claims", "whole_memory_wrong",
}

# These messages originate in the local validator, never from evidence/provider
# text. Unknown exceptions use a fixed fallback to keep retry feedback safe.
_SHARED_VALIDATION_FIELDS = {
    "Review memory version differs from injected version": "fingerprint",
    "Revision must change content and remain compact (<=12000 characters)": "new_content",
    "Revised memory must be source-blind": "new_content",
    "preserved_claims/revoked_claims must be lists": "preserved_claims",
    "Preserved claim must be an object": "preserved_claims",
    "Scope refinement must preserve claims and cannot revoke semantics": "preserved_claims",
    "Error correction needs explicit contradicted claims": "revoked_claims",
    "Revoked claim must be an object": "revoked_claims",
    "Whole-memory replacement must explicitly address the entire old content": "whole_memory_wrong",
    "Partial correction must preserve remaining valid claims": "preserved_claims",
}
for _label, _field in (
    ("problem_claim", "problem_claim"), ("new_content", "new_content"),
    ("preserved old claim", "preserved_claims"), ("preserved new claim", "preserved_claims"),
    ("revoked claim", "revoked_claims"), ("revocation reason", "revoked_claims"),
):
    for _suffix in ("must be nonempty text", "is not an exact source quote"):
        _SHARED_VALIDATION_FIELDS[f"{_label} {_suffix}"] = _field

_CITATION_MESSAGES = {
    "Evidence pointer must be an absolute JSON pointer",
    "Evidence pointer does not resolve",
    "Cite task, current trajectory or released feedback, not reviewer opinions",
    "Evidence references must point to scalar leaves",
    "evidence quote must be nonempty text",
    "evidence quote is not an exact source quote",
    "Revision requires current trajectory AND official feedback references",
}


def validate_memory_decision(
    decision: dict[str, Any],
    memory: MemoryItem,
    evidence: dict[str, Any],
    *,
    resolve_citation: Callable[[str], dict[str, Any]],
    delivered_ids: set[str],
) -> dict[str, Any]:
    """Validate one selected memory and return an ``apply_revisions`` decision.

    ``resolve_citation`` returns ``{pointer, quote}`` from the supplied archive;
    those references are independently checked against ``evidence``. The caller
    owns delivery tracking, retry/read budgets and atomic multi-memory apply.
    """
    ident = memory.memory_id

    def fail(code: str, field: str, message: str) -> None:
        raise ReviewValidationError(code, field, message, ident)

    def explanation(value: Any, field: str) -> str:
        if isinstance(value, list) and value and all(
            isinstance(part, str) and part.strip() for part in value
        ):
            # Preserve every supplied string, including its whitespace/order.
            return "\n".join(value)
        if not isinstance(value, str) or not value.strip():
            fail("invalid_text", field, "Provide nonempty text or a nonempty list of nonempty strings.")
        return value

    def references(value: Any, field: str, *, required: bool) -> list[dict[str, str]]:
        if not isinstance(value, list):
            fail("invalid_evidence_ids", field, "evidence_ids must be a list of delivered evidence IDs.")
        resolved = []
        for index, evidence_id in enumerate(value):
            item_field = f"{field}[{index}]"
            if not isinstance(evidence_id, str) or not evidence_id.strip():
                fail("invalid_evidence_id", item_field, "Each evidence ID must be nonempty text.")
            if evidence_id not in delivered_ids:
                fail("undelivered_evidence_id", item_field, "This evidence ID has not been delivered to this review.")
            try:
                ref = resolve_citation(evidence_id)
            except Exception:
                raise ReviewValidationError(
                    "unresolved_evidence_id", item_field,
                    "This evidence ID cannot be resolved from the supplied evidence.", ident,
                ) from None
            if not isinstance(ref, dict) or not isinstance(ref.get("pointer"), str) or not isinstance(ref.get("quote"), str):
                fail("invalid_citation", item_field, "Resolved evidence must contain a text pointer and quote.")
            resolved.append({"pointer": ref["pointer"], "quote": ref["quote"]})
        try:
            _validate_evidence(resolved, evidence, required=required)
        except (ValueError, TypeError) as exc:
            message = str(exc)
            if message not in _CITATION_MESSAGES:
                message = "Resolved evidence does not satisfy the archived evidence contract."
            raise ReviewValidationError("invalid_citation", field, message, ident) from None
        return resolved

    if not isinstance(decision, dict):
        fail("invalid_review", "review", "Each review must be a JSON object.")
    if set(decision) - _FIELDS:
        fail("unknown_fields", "review", "Unexpected review fields; use evidence_ids instead of raw evidence references.")
    if decision.get("memory_id") != ident:
        fail("memory_id_mismatch", "memory_id", "Review must name the currently selected memory.")
    fingerprint = memory_identity(memory)["fingerprint"]
    if "fingerprint" in decision and decision["fingerprint"] != fingerprint:
        fail("fingerprint_mismatch", "fingerprint", "An optional fingerprint must match the selected memory version.")
    action = decision.get("action")
    if not isinstance(action, str) or action not in EDIT_ACTIONS | {"keep", "uncertain"}:
        fail("invalid_action", "action", "Use keep, uncertain, refine_scope, or correct_error.")
    for field in ("safety_effect", "goal_effect"):
        value = decision.get(field)
        if not isinstance(value, str) or value not in EFFECTS:
            fail("invalid_effect", field, "Use helpful, harmful, neutral, or unknown.")
    new_content = decision.get("new_content")
    if not isinstance(new_content, str) or not new_content.strip():
        fail("invalid_new_content", "new_content", "Every action requires the complete final memory text.")
    if action not in EDIT_ACTIONS and new_content != memory.content:
        fail("keep_content_changed", "new_content", "For keep/uncertain, copy the original memory content byte for byte.")

    normalized = deepcopy(decision)
    normalized["fingerprint"] = fingerprint
    for field in ("reason", "alternative_explanations"):
        normalized[field] = explanation(decision.get(field), field)
    normalized["evidence"] = references(
        normalized.pop("evidence_ids", None), "evidence_ids", required=action in EDIT_ACTIONS,
    )
    if "revoked_claims" in normalized:
        revoked = normalized["revoked_claims"]
        if not isinstance(revoked, list):
            fail("invalid_revoked_claims", "revoked_claims", "revoked_claims must be a list.")
        for index, claim in enumerate(revoked):
            field = f"revoked_claims[{index}]"
            if not isinstance(claim, dict):
                fail("invalid_revoked_claim", field, "Each revoked claim must be an object.")
            if set(claim) - {"quote", "reason", "evidence_ids"}:
                fail("unknown_fields", field, "Revoked claims accept only quote, reason, and evidence_ids.")
            claim["reason"] = explanation(claim.get("reason"), f"{field}.reason")
            claim["evidence"] = references(
                claim.pop("evidence_ids", None), f"{field}.evidence_ids", required=action == "correct_error",
            )
    if action not in EDIT_ACTIONS:
        normalized["new_content"] = None
    try:
        return validate_normalized_decision(normalized, evidence, memory)
    except (ValueError, TypeError) as exc:
        message = str(exc)
        field = _SHARED_VALIDATION_FIELDS.get(message, "review")
        if message not in _SHARED_VALIDATION_FIELDS:
            message = "Decision does not satisfy the memory review contract."
        raise ReviewValidationError("semantic_contract", field, message, ident) from None
