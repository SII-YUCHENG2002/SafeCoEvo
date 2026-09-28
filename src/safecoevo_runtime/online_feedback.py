"""Official-feedback state and checkpoints for online Harness evolution.

A source adapter supplies the benchmark outcome. This module normalizes it and
records a versioned delta; Memory updates are authored through artifact evolution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .contracts import MemoryItem, PermissionExperienceItem, TaskOutcome
from .dataset import SafeCoEvoCase
from .memory import InMemoryMemoryStore
from .artifact_patcher import ArtifactState
from .permission import InMemoryPermissionExperienceStore
from .runtime import HarnessSession


OutcomeLabel = Literal["passed", "failed", "not_assessed"]


def _outcome_label(*, unsafe: bool | None, inverse: bool) -> OutcomeLabel:
    if not isinstance(unsafe, bool):
        return "not_assessed"
    value = not unsafe if inverse else unsafe
    return "passed" if value else "failed"


def _retrieval_semantics(config: dict[str, Any]) -> dict[str, Any]:
    """Remove transport-only knobs from resumability checks."""

    ignored = {"timeout_seconds", "max_retries", "retry_backoff_seconds"}
    return {key: value for key, value in config.items() if key not in ignored}


@dataclass(frozen=True)
class OfficialEpisodeFeedback:
    """Source-neutral ground-truth feedback available after one episode."""

    task_id: str
    harness_version_before: str
    safety: OutcomeLabel
    goal: OutcomeLabel
    evaluation_complete: bool
    evaluator_name: str | None
    evaluator_kind: str
    official_signals: dict[str, Any]
    availability: dict[str, bool]
    trace_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HarnessDelta:
    """Recorded state transition for one completed episode."""

    parent_version: str
    next_version: str
    task_id: str
    change_type: str
    memory_id: str | None
    rationale: str
    evidence_trace_hashes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence_trace_hashes"] = list(self.evidence_trace_hashes)
        return value


def official_episode_feedback(
    *,
    case: SafeCoEvoCase,
    outcome: TaskOutcome,
    session: HarnessSession,
    harness_version: str,
) -> OfficialEpisodeFeedback:
    """Normalize a native oracle without exposing a source name to the target."""

    signals = dict(outcome.metadata.get("official_signals") or {})
    if outcome.unsafe_outcome is not None:
        signals.setdefault("unsafe_outcome", outcome.unsafe_outcome)
    if outcome.legitimate_task_success is not None:
        signals.setdefault("task_success", outcome.legitimate_task_success)
    verdicts = list(session.state.guard_verdicts.values())
    return OfficialEpisodeFeedback(
        task_id=case.public.task_id,
        harness_version_before=harness_version,
        # unsafe_outcome=True means the safety objective failed.
        safety=_outcome_label(unsafe=outcome.unsafe_outcome, inverse=True),
        goal=_outcome_label(unsafe=outcome.legitimate_task_success, inverse=False),
        evaluation_complete=outcome.verified,
        evaluator_name=outcome.verifier_name,
        evaluator_kind="native_oracle" if outcome.verified else "pending_or_unavailable",
        official_signals=signals,
        availability={
            "safety": isinstance(outcome.unsafe_outcome, bool),
            "goal": isinstance(outcome.legitimate_task_success, bool),
        },
        trace_summary={
            "turn_count": sum(message.get("role") == "assistant" for message in session.state.messages),
            "guard_observations": {
                "safe": sum(verdict.safety == "safe" for verdict in verdicts),
                "unsafe": sum(verdict.safety == "unsafe" for verdict in verdicts),
                "unknown": sum(verdict.safety == "unknown" for verdict in verdicts),
            },
        },
    )


@dataclass
class OnlineEvolutionState:
    """Shared state carried forward through a fixed ordered task stream."""

    initial_version: str = "H0"
    # Episodes only affect a future task when their verified lesson is
    # relevant to that task's public request or tool environment.
    memory_store: InMemoryMemoryStore = field(default_factory=InMemoryMemoryStore)
    permission_store: InMemoryPermissionExperienceStore = field(default_factory=InMemoryPermissionExperienceStore)
    artifact_state: ArtifactState = field(default_factory=lambda: ArtifactState(prompt_addendum=""))
    completed_episodes: int = 0
    feedback_history: list[OfficialEpisodeFeedback] = field(default_factory=list)
    deltas: list[HarnessDelta] = field(default_factory=list)
    # Endpoint/model metadata only: credentials never enter checkpoints.
    memory_retrieval_config: dict[str, Any] = field(default_factory=dict)

    @property
    def version(self) -> str:
        return self.initial_version if self.completed_episodes == 0 else f"{self.initial_version}.{self.completed_episodes}"

    def prepare_update(
        self,
        *,
        case: SafeCoEvoCase,
        outcome: TaskOutcome,
        session: HarnessSession,
    ) -> tuple[OfficialEpisodeFeedback, HarnessDelta]:
        """Record official feedback for the completed episode.

        Memory artifacts are authored through artifact evolution or curated
        seeds. Episode feedback is retained for audit and subsequent updates.
        """

        parent = self.version
        feedback = official_episode_feedback(
            case=case,
            outcome=outcome,
            session=session,
            harness_version=parent,
        )
        next_version = f"{self.initial_version}.{self.completed_episodes + 1}"
        hashes = tuple(str(value) for value in session.state.annotations.get("trace_hashes", ()) if value)
        delta = HarnessDelta(
            parent_version=parent,
            next_version=next_version,
            task_id=case.public.task_id,
            change_type="record_feedback_without_auto_memory",
            memory_id=None,
            rationale=(
                "Official post-episode feedback was recorded. Automatic feedback-to-memory writing is disabled; "
                "memory updates must come from explicit artifact evolution or curated seed artifacts."
            ),
            evidence_trace_hashes=hashes[-3:],
        )
        return feedback, delta

    def configure_memory_retrieval(self, config: dict[str, Any]) -> None:
        """Bind a resumable stream to the retrieval semantics it actually uses."""

        if not isinstance(config, dict) or not config:
            raise ValueError("Memory retrieval configuration must be a non-empty mapping")
        normalized = dict(config)
        if normalized.get("mode") != "lexical":
            raise ValueError("Only lexical Memory retrieval is supported")
        if self.memory_retrieval_config and _retrieval_semantics(self.memory_retrieval_config) != _retrieval_semantics(normalized):
            raise ValueError("Cannot resume an online stream with different memory retrieval semantics")
        self.memory_retrieval_config = normalized

    def commit_update(self, feedback: OfficialEpisodeFeedback, delta: HarnessDelta) -> None:
        """Advance only after the task trace and MemoryWriter have completed."""

        if delta.parent_version != self.version:
            raise ValueError("Online update parent version does not match current stream state")
        self.feedback_history.append(feedback)
        self.deltas.append(delta)
        self.completed_episodes += 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize only verified state needed to resume the ordered stream."""

        return {
            "schema_version": 3,
            "initial_version": self.initial_version,
            "completed_episodes": self.completed_episodes,
            "feedback_history": [item.to_dict() for item in self.feedback_history],
            "deltas": [item.to_dict() for item in self.deltas],
            "memory_items": [asdict(item) for item in self.memory_store.items],
            "permission_items": [asdict(item) for item in self.permission_store.items],
            "memory_retrieval_config": dict(self.memory_retrieval_config),
            "artifact_state": {
                "version": self.artifact_state.version,
                "prompt_addendum": self.artifact_state.prompt_addendum,
                "guard_policy": dict(self.artifact_state.guard_policy),
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OnlineEvolutionState":
        """Restore a checkpoint, validating that its version lineage is sound."""

        if not isinstance(value, dict) or value.get("schema_version") != 3:
            raise ValueError("Unsupported online state schema; start a fresh output directory")
        initial_version = value.get("initial_version")
        completed = value.get("completed_episodes")
        raw_feedback = value.get("feedback_history")
        raw_deltas = value.get("deltas")
        raw_memory = value.get("memory_items")
        raw_permission = value.get("permission_items", [])
        if not isinstance(initial_version, str) or not initial_version:
            raise ValueError("Checkpoint needs an initial_version")
        if not isinstance(completed, int) or isinstance(completed, bool) or completed < 0:
            raise ValueError("Checkpoint has invalid completed_episodes")
        if not all(isinstance(items, list) for items in (raw_feedback, raw_deltas, raw_memory, raw_permission)):
            raise ValueError("Checkpoint history fields must be lists")
        feedback = [_feedback_from_dict(item) for item in raw_feedback]
        deltas = [_delta_from_dict(item) for item in raw_deltas]
        memory = [_memory_item_from_dict(item) for item in raw_memory]
        permission = [_permission_item_from_dict(item) for item in raw_permission]
        if completed != len(feedback) or completed != len(deltas):
            raise ValueError("Checkpoint episode count does not match its histories")
        expected_parent = initial_version
        for item in deltas:
            if item.parent_version != expected_parent:
                raise ValueError("Checkpoint delta lineage is discontinuous")
            expected_parent = item.next_version
        state = cls(initial_version=initial_version)
        raw_artifact_state = value.get("artifact_state")
        if isinstance(raw_artifact_state, dict):
            guard_policy = raw_artifact_state.get("guard_policy", {})
            if not isinstance(guard_policy, dict):
                raise ValueError("Checkpoint artifact_state.guard_policy must be an object")
            state.artifact_state = ArtifactState(
                prompt_addendum=str(raw_artifact_state.get("prompt_addendum", "")),
                guard_policy=dict(guard_policy),
                version=str(raw_artifact_state.get("version", "A0")),
            )
        state.completed_episodes = completed
        state.feedback_history = feedback
        state.deltas = deltas
        for item in memory:
            state.memory_store.append(item)
        for item in permission:
            state.permission_store.append(item)
        raw_retrieval = value.get("memory_retrieval_config", {})
        if not isinstance(raw_retrieval, dict):
            raise ValueError("Checkpoint memory retrieval configuration is invalid")
        state.memory_retrieval_config = dict(raw_retrieval)
        return state


def _feedback_from_dict(value: Any) -> OfficialEpisodeFeedback:
    if not isinstance(value, dict):
        raise ValueError("Checkpoint feedback must be an object")
    required = ("task_id", "harness_version_before", "safety", "goal", "evaluation_complete", "evaluator_kind")
    if any(field not in value for field in required):
        raise ValueError("Checkpoint feedback is incomplete")
    safety = value["safety"]
    goal = value["goal"]
    if safety not in {"passed", "failed", "not_assessed"} or goal not in {"passed", "failed", "not_assessed"}:
        raise ValueError("Checkpoint feedback has invalid labels")
    if not isinstance(value["task_id"], str) or not isinstance(value["harness_version_before"], str):
        raise ValueError("Checkpoint feedback has invalid identifiers")
    if not isinstance(value["evaluation_complete"], bool) or not isinstance(value["evaluator_kind"], str):
        raise ValueError("Checkpoint feedback has invalid evaluation metadata")
    signals = value.get("official_signals", {})
    availability = value.get("availability", {})
    trace_summary = value.get("trace_summary", {})
    if not all(isinstance(item, dict) for item in (signals, availability, trace_summary)):
        raise ValueError("Checkpoint feedback mappings are invalid")
    evaluator_name = value.get("evaluator_name")
    if evaluator_name is not None and not isinstance(evaluator_name, str):
        raise ValueError("Checkpoint evaluator_name must be text or null")
    return OfficialEpisodeFeedback(
        task_id=value["task_id"],
        harness_version_before=value["harness_version_before"],
        safety=safety,
        goal=goal,
        evaluation_complete=value["evaluation_complete"],
        evaluator_name=evaluator_name,
        evaluator_kind=value["evaluator_kind"],
        official_signals=dict(signals),
        availability=dict(availability),
        trace_summary=dict(trace_summary),
    )


def _delta_from_dict(value: Any) -> HarnessDelta:
    if not isinstance(value, dict):
        raise ValueError("Checkpoint delta must be an object")
    required = ("parent_version", "next_version", "task_id", "change_type", "rationale", "evidence_trace_hashes")
    if any(field not in value for field in required):
        raise ValueError("Checkpoint delta is incomplete")
    hashes = value["evidence_trace_hashes"]
    if not all(isinstance(field, str) for field in (value["parent_version"], value["next_version"], value["task_id"], value["change_type"], value["rationale"])):
        raise ValueError("Checkpoint delta has invalid text fields")
    if not isinstance(hashes, list) or not all(isinstance(item, str) for item in hashes):
        raise ValueError("Checkpoint delta hashes are invalid")
    memory_id = value.get("memory_id")
    if memory_id is not None and not isinstance(memory_id, str):
        raise ValueError("Checkpoint delta memory_id must be text or null")
    return HarnessDelta(
        parent_version=value["parent_version"],
        next_version=value["next_version"],
        task_id=value["task_id"],
        change_type=value["change_type"],
        memory_id=memory_id,
        rationale=value["rationale"],
        evidence_trace_hashes=tuple(hashes),
    )


def _memory_item_from_dict(value: Any) -> MemoryItem:
    if not isinstance(value, dict):
        raise ValueError("Checkpoint memory item must be an object")
    required = ("memory_id", "content", "evidence_trace_hashes", "verified", "tags")
    if any(field not in value for field in required):
        raise ValueError("Checkpoint memory item is incomplete")
    hashes = value["evidence_trace_hashes"]
    tags = value["tags"]
    if not isinstance(hashes, (list, tuple)) or not isinstance(tags, (list, tuple)):
        raise ValueError("Checkpoint memory sequence fields are invalid")
    return MemoryItem(
        memory_id=str(value["memory_id"]),
        content=str(value["content"]),
        evidence_trace_hashes=tuple(str(item) for item in hashes),
        verified=bool(value["verified"]),
        tags=tuple(str(item) for item in tags),
    )


def _permission_item_from_dict(value: Any) -> PermissionExperienceItem:
    if not isinstance(value, dict):
        raise ValueError("Checkpoint permission item must be an object")
    required = ("permission_id", "content", "evidence_trace_hashes", "verified", "tags")
    if any(field not in value for field in required):
        raise ValueError("Checkpoint permission item is incomplete")
    hashes = value["evidence_trace_hashes"]
    tags = value["tags"]
    if not isinstance(hashes, (list, tuple)) or not isinstance(tags, (list, tuple)):
        raise ValueError("Checkpoint permission sequence fields are invalid")
    return PermissionExperienceItem(
        permission_id=str(value["permission_id"]),
        content=str(value["content"]),
        evidence_trace_hashes=tuple(str(item) for item in hashes),
        verified=bool(value["verified"]),
        tags=tuple(str(item) for item in tags),
    )
