"""Stable contracts for the evolvable Safety Harness runtime.

These contracts are independent from benchmark-specific runners. A source
executor can adapt a benchmark to these types without
revealing its private evaluator payload to the target agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class Hook(str, Enum):
    """The six fixed processor boundaries in a SafeCoEvo session."""

    TASK_START = "task_start"
    # A request-local boundary. Processors may enrich the next target-model
    # request without mutating the durable conversation or native tool table.
    BEFORE_MODEL = "before_model"
    AFTER_MODEL = "after_model"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    TASK_END = "task_end"


class Resource(str, Enum):
    """Resources a processor may declare that it reads or writes."""

    PROMPT = "prompt"
    MEMORY = "memory"
    SKILL = "skill"
    TOOL = "tool"
    PERMISSION = "permission"
    CONTEXT = "context"
    ACTION = "action"
    SAFETY_OBSERVATION = "safety_observation"
    TRACE = "trace"
    EVOLUTION = "evolution"


SafetyLabel = Literal["safe", "unsafe", "unknown"]
ActionKind = Literal["tool", "final"]


@dataclass(frozen=True)
class ToolCall:
    """A source-neutral, public tool request proposed by the target agent."""

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class CandidateAction:
    """One target-agent output captured before it affects the environment."""

    action_id: str
    kind: ActionKind
    text: str = ""
    tool_call: ToolCall | None = None

    def __post_init__(self) -> None:
        if self.kind == "tool" and self.tool_call is None:
            raise ValueError("Tool candidate actions require a ToolCall")
        if self.kind == "final" and self.tool_call is not None:
            raise ValueError("Final candidate actions cannot carry a ToolCall")


@dataclass(frozen=True)
class GuardVerdict:
    """Normalized Guard output; raw model output remains in the trace only."""

    safety: SafetyLabel
    risk_type: str = "unknown"
    risk_level: str = "unknown"
    reason: str = ""
    evidence: tuple[str, ...] = ()
    availability: str = "available"
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.safety not in {"safe", "unsafe", "unknown"}:
            raise ValueError(f"Unsupported safety label: {self.safety!r}")


@dataclass(frozen=True)
class TaskOutcome:
    """Source-adapter outcome supplied only after a task has ended."""

    status: str
    unsafe_outcome: bool | None = None
    legitimate_task_success: bool | None = None
    verifier_name: str | None = None
    verified: bool = False
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class MemoryItem:
    """A verified, reusable safety experience rather than raw conversation text.

    Only verified items are eligible for task-end commit and retrieval.
    """

    memory_id: str
    content: str
    evidence_trace_hashes: tuple[str, ...]
    verified: bool
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PermissionExperienceItem:
    """A high-level authorization lesson injected as advisory context."""

    permission_id: str
    content: str
    evidence_trace_hashes: tuple[str, ...]
    verified: bool
    tags: tuple[str, ...] = ()
