"""Shared tool-call and Guard diagnosis schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


SafetyLabel = Literal["safe", "unsafe", "unknown"]
RiskType = Literal[
    "privacy_leakage",
    "prompt_injection",
    "dangerous_tool",
    "memory_poisoning",
    "unauthorized_action",
    "none",
    "unknown",
]
@dataclass(slots=True)
class ToolCallRecord:
    """A proposed or executed tool call."""

    tool_name: str
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrajectoryStep:
    """One step in a tool-using agent trajectory."""

    step_id: int
    model_output: str = ""
    tool_call: ToolCallRecord | None = None
    tool_result: str | None = None
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SafetyDiagnosis:
    """Structured output expected from the safety diagnoser."""

    safety: SafetyLabel
    risk_type: RiskType = "none"
    risk_level: Literal["low", "medium", "high", "critical", "none", "unknown"] = "none"
    unsafe_step: int | None = None
    violated_rule: str = ""
    reason: str = ""
    risk_feedback_message: str = ""
    revision_guidance: str = ""
    raw_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
