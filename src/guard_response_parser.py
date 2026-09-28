"""Robust parsing for safety diagnoser JSON outputs."""

from __future__ import annotations

import json
import re
from typing import Any

from trajectory_schema import SafetyDiagnosis


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

_SAFETY_VALUES = {"safe", "unsafe", "unknown"}
_RISK_VALUES = {
    "privacy_leakage",
    "prompt_injection",
    "dangerous_tool",
    "memory_poisoning",
    "unauthorized_action",
    "none",
    "unknown",
}
_RISK_LEVEL_VALUES = {"low", "medium", "high", "critical", "none", "unknown"}


def _extract_json_candidate(text: str) -> str:
    """Extract the first likely JSON object from a model response."""

    stripped = text.strip()
    fence = _FENCE_RE.search(stripped)
    if fence:
        stripped = fence.group(1).strip()

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    if start < 0:
        raise ValueError("No JSON object start found in diagnoser output")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(stripped)):
        char = stripped[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : idx + 1]

    raise ValueError("No complete JSON object found in diagnoser output")


def _normalize_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _normalize_choice(value: Any, allowed: set[str], default: str) -> str:
    normalized = _normalize_string(value, default=default).lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    return normalized if normalized in allowed else default


def _normalize_safety(value: Any) -> str:
    """Accept the string and boolean forms emitted by supported Guard models."""

    # AgentDoG's OpenAI-compatible endpoint emits JSON booleans, while the
    # AgentDoG may express the same decision as explicit strings.
    if isinstance(value, bool):
        return "safe" if value else "unsafe"
    return _normalize_choice(value, _SAFETY_VALUES, "unknown")


def _normalize_unsafe_step(value: Any) -> int | None:
    if value in (None, "", "none", "null"):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def parse_safety_diagnosis(text: str) -> SafetyDiagnosis:
    """Parse model text into a normalized SafetyDiagnosis."""

    candidate = _extract_json_candidate(text)
    data = json.loads(candidate)
    if not isinstance(data, dict):
        raise ValueError("Diagnoser JSON must be an object")

    safety = _normalize_safety(data.get("safety"))
    risk_type = _normalize_choice(data.get("risk_type"), _RISK_VALUES, "unknown")
    risk_level = _normalize_choice(data.get("risk_level"), _RISK_LEVEL_VALUES, "unknown")

    if safety == "safe":
        risk_type = "none" if risk_type == "unknown" else risk_type
        risk_level = "none" if risk_level == "unknown" else risk_level

    return SafetyDiagnosis(
        safety=safety,  # type: ignore[arg-type]
        risk_type=risk_type,  # type: ignore[arg-type]
        risk_level=risk_level,  # type: ignore[arg-type]
        unsafe_step=_normalize_unsafe_step(data.get("unsafe_step")),
        violated_rule=_normalize_string(data.get("violated_rule")),
        reason=_normalize_string(data.get("reason")),
        raw_output=text,
        metadata={"parsed_json": data},
    )


def safe_parse_safety_diagnosis(text: str) -> SafetyDiagnosis:
    """Parse diagnosis output; return an unknown diagnosis on parser failure."""

    try:
        return parse_safety_diagnosis(text)
    except Exception as exc:  # intentionally broad: parser must not crash guard runner
        return SafetyDiagnosis(
            safety="unknown",
            risk_type="unknown",
            risk_level="unknown",
            unsafe_step=None,
            violated_rule="",
            reason=f"Failed to parse safety diagnoser output: {type(exc).__name__}: {exc}",
            raw_output=text,
            metadata={"parse_error": str(exc)},
        )
