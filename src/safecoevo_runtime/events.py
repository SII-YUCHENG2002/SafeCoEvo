"""Event and state objects for the new Safety Harness lifecycle."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any

from .contracts import CandidateAction, GuardVerdict, Hook, MemoryItem, TaskOutcome


@dataclass(frozen=True)
class PublicTask:
    """The only task payload exposed to the target agent and its processors."""

    task_id: str
    system: dict[str, Any]
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...]
    runtime: dict[str, Any]

    def agent_payload(self) -> dict[str, Any]:
        """Return a deep copy so callers cannot mutate the source dataset record."""

        return copy.deepcopy(
            {
                "task_id": self.task_id,
                "system": self.system,
                "messages": list(self.messages),
                "tools": list(self.tools),
                "runtime": self.runtime,
            }
        )


@dataclass
class HarnessState:
    """Mutable per-task state. Private evaluator data is intentionally absent."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    formatted_messages: list[dict[str, Any]] = field(default_factory=list)
    retrieved_memory: list[MemoryItem] = field(default_factory=list)
    pending_memory: list[MemoryItem] = field(default_factory=list)
    guard_requests: dict[str, Any] = field(default_factory=dict)
    guard_verdicts: dict[str, GuardVerdict] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)
    # Generated Processors may expose only informational observations through
    # this buffer. The session attaches them at protocol-safe boundaries.
    processor_observations: list[dict[str, Any]] = field(default_factory=list)
    evolution_signals: list[dict[str, Any]] = field(default_factory=list)
    evolution_evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class HarnessEvent:
    """One lifecycle event passed through processors registered for its Hook."""

    hook: Hook
    task: PublicTask
    state: HarnessState
    action: CandidateAction | None = None
    # ``proposed_action`` is immutable evidence of what the target model
    # emitted. ``action`` is the current effective action seen by downstream
    # processors and may be replaced by an intervention processor.
    proposed_action: CandidateAction | None = None
    execution_status: str = "pending"
    synthetic_tool_result: str | None = None
    intercepted_by: tuple[str, ...] = ()
    tool_result: Any = None
    outcome: TaskOutcome | None = None
    # ``before_model`` processors edit only these ephemeral request values.
    # They are never written back to ``state.formatted_messages``.
    model_messages: list[dict[str, Any]] | None = None
    available_tools: list[dict[str, Any]] | None = None
    sequence: int = 0

    def with_effective_action(self, action: CandidateAction) -> "HarnessEvent":
        """Return an event whose downstream action differs from the proposal."""

        original = self.proposed_action or self.action
        if original is not None and action.kind != original.kind:
            raise ValueError(
                "A Processor may transform an action only within its original kind; "
                "one model turn is either a final answer or one tool call."
            )
        return replace(
            self,
            action=action,
            proposed_action=original,
            execution_status="modified",
        )

    def with_execution_decision(
        self,
        *,
        approved: bool,
        synthetic_result: str | None = None,
        processor_id: str | None = None,
    ) -> "HarnessEvent":
        """Return an event that records an explicit execution decision.

        Action-hook processors use this transformation to record whether an
        action may execute. The raw proposed action remains available.
        """

        return replace(
            self,
            proposed_action=self.proposed_action or self.action,
            execution_status="approved" if approved else "withheld",
            synthetic_tool_result=synthetic_result,
            intercepted_by=(
                *self.intercepted_by,
                *(() if processor_id is None else (processor_id,)),
            ),
            tool_result=self.tool_result,
            outcome=self.outcome,
            sequence=self.sequence,
        )

    def with_internal_tool_result(
        self,
        *,
        result: str,
        processor_id: str,
    ) -> "HarnessEvent":
        """Mark a Harness-owned tool call as completed without native execution."""

        return replace(
            self,
            proposed_action=self.proposed_action or self.action,
            execution_status="internal",
            synthetic_tool_result=result,
            intercepted_by=(*self.intercepted_by, processor_id),
        )


def agent_visible_observations(state: HarnessState) -> dict[str, dict[str, object]]:
    """Return prior Guard observations as data rather than behavioral commands."""

    return {
        action_id: {
            "safety": verdict.safety,
            "risk_type": verdict.risk_type,
            "risk_level": verdict.risk_level,
            "reason": verdict.reason,
            "evidence": list(verdict.evidence),
            "availability": verdict.availability,
        }
        for action_id, verdict in state.guard_verdicts.items()
    }


HARNESS_MESSAGE_METADATA_PREFIX = "_harness_"


def strip_harness_message_metadata(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove runtime-only provenance before messages leave the Harness."""

    return [
        {
            str(key): copy.deepcopy(value)
            for key, value in message.items()
            if not str(key).startswith(HARNESS_MESSAGE_METADATA_PREFIX)
        }
        for message in messages
    ]


def agent_visible_messages(
    state: HarnessState,
    *,
    include_harness_metadata: bool = False,
) -> list[dict[str, object]]:
    """Return the exact protocol-valid context delivered to target and Guard.

    Per-action Guard observations are attached to their matching ``tool``
    result in ``HarnessSession.record_tool_result``. Keeping the standard
    assistant-tool-assistant sequence avoids malformed function-calling
    histories while preserving every earlier verdict in its original context.
    """

    messages = copy.deepcopy(state.formatted_messages)
    if include_harness_metadata:
        return messages
    return strip_harness_message_metadata(messages)
