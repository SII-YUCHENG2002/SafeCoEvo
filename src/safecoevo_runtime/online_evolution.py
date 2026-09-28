"""Post-episode controller for fixed artifact updates."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import SafeCoEvoCase
from .artifact_patcher import ArtifactState, apply_artifact_patch_to_runtime, normalize_artifact_patch, validate_artifact_patch
from .memory import MemoryStore
from .permission import PermissionExperienceStore
from .online_feedback import OfficialEpisodeFeedback
from .processors import Guard
from .runtime import HarnessSession
from .skills import MIN_SKILL_CREATE_EPISODES, SkillRegistry, load_skill_manifest
from .evolution_graph import (
    EvolutionAgent,
    EvolutionDeployment,
    EvolutionPolicy,
    ProcessorGraph,
    advance_graph_version,
    evolution_policy,
)
from .evolution_workspace import EvolutionWorkspace, EvolutionWorkspaceTools, create_evolution_workspace
from .evolution_privacy import (
    contains_source_identity,
    source_blind_feedback,
    source_blind_graph_view,
    source_blind_trace,
)


@dataclass(frozen=True)
class BatchEvolutionEpisode:
    """One completed episode carried into a later batch-level evolution call."""

    case: SafeCoEvoCase
    session: HarnessSession
    feedback: OfficialEpisodeFeedback
    trace_records: list[dict[str, Any]]
    next_version: str


def _compact_guard_verdicts(value: Any) -> Any:
    """Keep Guard decisions without repeating full Guard prompts or long rationales."""

    if not isinstance(value, dict):
        return copy.deepcopy(value)
    out: dict[str, Any] = {}
    for key, verdict in value.items():
        if not isinstance(verdict, dict):
            out[key] = copy.deepcopy(verdict)
            continue
        item = {
            field: copy.deepcopy(verdict.get(field))
            for field in ("safety", "risk_type", "risk_level", "availability", "parser")
            if field in verdict
        }
        evidence = verdict.get("evidence")
        if isinstance(evidence, list):
            item["evidence_count"] = len(evidence)
        reason = verdict.get("reason")
        if isinstance(reason, str) and reason.strip():
            item["reason_excerpt"] = reason[:800]
            item["reason_bytes"] = len(reason.encode("utf-8"))
        out[key] = item
    return out


def _full_public_trace(trace_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a direct, complete episode trajectory without repeated snapshots.

    The append-only harness trace records are cumulative snapshots: every record
    repeats the full model-visible conversation so far. Passing those snapshots
    verbatim can turn a short episode into a multi-megabyte Evolution request.
    For the model-facing evidence we keep the complete final conversation and a
    per-record action/Guard/event timeline, while the raw trace remains on disk
    for audit.
    """

    if not trace_records:
        return []
    final = copy.deepcopy(trace_records[-1])
    event_log: list[dict[str, Any]] = []
    hash_chain: list[dict[str, Any]] = []
    for raw in trace_records:
        event = {
            "sequence": raw.get("sequence"),
            "hook": raw.get("hook"),
            "proposed_action": copy.deepcopy(raw.get("proposed_action")),
            "effective_action": copy.deepcopy(raw.get("effective_action")),
            "execution_status": copy.deepcopy(raw.get("execution_status")),
            "intercepted_by": copy.deepcopy(raw.get("intercepted_by")),
            "guard_verdicts": _compact_guard_verdicts(raw.get("guard_verdicts")),
            # Full Guard prompts duplicate the model-visible messages and can
            # dominate the Evolution request. Guard decisions are retained via
            # guard_verdicts and final_agent_messages.
            "guard_request_count": len(raw.get("guard_requests") or []),
            "processor_observations": copy.deepcopy(raw.get("processor_observations")),
            "tool_result": copy.deepcopy(raw.get("tool_result")),
            "synthetic_tool_result": copy.deepcopy(raw.get("synthetic_tool_result")),
        }
        event = {key: value for key, value in event.items() if value not in (None, [], {})}
        if event:
            event_log.append(event)
        hash_chain.append(
            {
                "sequence": raw.get("sequence"),
                "hook": raw.get("hook"),
                "entry_hash": raw.get("entry_hash"),
                "previous_hash": raw.get("previous_hash"),
            }
        )
    final_annotations = final.get("annotations") if isinstance(final.get("annotations"), dict) else {}
    trajectory = {
        "schema_version": 2,
        "view": "complete_episode_trajectory_without_repeated_snapshots",
        "record_count": len(trace_records),
        "final_agent_messages": copy.deepcopy(final.get("agent_messages", [])),
        "available_tools": copy.deepcopy(final.get("available_tools", [])),
        "event_log": event_log,
        "final_outcome": copy.deepcopy(final.get("outcome")),
        "final_harness_annotations": {
            key: copy.deepcopy(final_annotations.get(key))
            for key in (
                "harness_version",
                "memory_retrieval",
                "permission_experience",
                "skill_retrieval",
                "skill_usage",
                "guard_observation_summaries",
                "guard_preflights",
                "action_executions",
                "trace_hashes",
            )
            if key in final_annotations
        },
        "raw_record_hash_chain": hash_chain,
        "raw_snapshot_note": "Raw append-only snapshot records are retained on disk for audit; this direct view removes repeated cumulative message snapshots without omitting the final conversation or action timeline.",
    }
    return [trajectory]


def _version_number(value: str, *, prefix: str) -> int:
    """Parse the monotonically increasing H/S version names used on disk."""

    suffix = value[len(prefix):] if value.startswith(prefix) else ""
    if not suffix or not all(part.isdigit() for part in suffix.split(".")):
        raise ValueError(f"Expected {prefix}<number> version, got {value!r}")
    return int(suffix.split(".")[-1])


def _workspace_task_specific_literals(workspace: EvolutionWorkspace) -> tuple[str, ...]:
    """Read identifiers that cannot appear verbatim in a reusable Skill."""

    try:
        value = json.loads((workspace.root / "episode" / "task.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    if not isinstance(value, dict):
        return ()
    values: list[str] = []
    if isinstance(value.get("task_id"), str):
        values.append(value["task_id"])
    tools = value.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            if isinstance(name, str):
                values.append(name)
    return tuple(values)


@dataclass
class OnlineGraphEvolutionController:
    """Owns active graph state and deploys at most one candidate per episode.

    ``evolve`` is intentionally called only after ``HarnessSession.end``. The
    active graph is immutable for the life of a session; the controller returns
    a successor graph that the caller can use only when constructing the next
    session.
    """

    graph: ProcessorGraph
    agent: EvolutionAgent
    root: Path
    guard: Guard
    memory_store: MemoryStore
    permission_store: PermissionExperienceStore | None = None
    artifact_state: ArtifactState | None = None
    policy: EvolutionPolicy | None = None
    skill_registry: SkillRegistry | None = None
    skill_evolution_interval: int = 1
    # Set by the runner for one episode when the periodic skill-consolidation
    # round owns skill changes; per-episode skill file_updates are then rejected
    # gracefully (status=rejected) instead of double-writing the registry.
    skill_updates_locked: bool = False
    # Opt-in independent reviewer owns existing-memory revisions only.
    protect_existing_memory: bool = False

    def initialize(self) -> None:
        if self.policy is None:
            self.policy = evolution_policy("fixed_artifact_patch")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.artifact_state is None:
            self.artifact_state = ArtifactState(prompt_addendum="")
        self._write_json(self.root / "versions" / self.graph.version / "graph.json", self.graph.to_dict())
        self._write_json(self.root / "policy.json", self.policy.to_dict())
        if self.skill_registry is not None:
            self.skill_registry.write(self.root.parent / "skills" / "versions" / self.skill_registry.version / "registry.json")
        if self.skill_evolution_interval < 1:
            raise ValueError("Skill evolution interval must be positive")


    def evolve(
        self,
        *,
        case: SafeCoEvoCase,
        session: HarnessSession,
        feedback: OfficialEpisodeFeedback,
        trace_records: list[dict[str, Any]],
        next_version: str,
    ) -> EvolutionDeployment:
        """Ask the agent for a candidate, validate it, then atomically switch H."""

        if session.harness.config.version != self.graph.version:
            raise ValueError("Session graph version does not match active evolution graph")
        # The coordinator keeps the real graph and evaluator provenance.  The
        # Evolution Agent receives only this reversible, source-blind view.
        graph_view = source_blind_graph_view(self.graph)
        evidence = build_online_evidence(
            case=case,
            session=session,
            feedback=feedback,
            trace_records=trace_records,
            processor_id_aliases=graph_view.actual_to_view_processor_id,
        )
        episode_number = _version_number(next_version, prefix="H")
        # Keep enough source-blind history available for the system-prompt Skill
        # evidence rule without exposing a separate Skill-specific evolution
        # payload in the user evidence.
        skill_review_window = max(self.skill_evolution_interval, MIN_SKILL_CREATE_EPISODES)
        hashes = tuple(str(value) for value in session.state.annotations.get("trace_hashes", ()) if value)
        workspace = create_evolution_workspace(
            evolution_root=self.root,
            task_id=case.public.task_id,
            graph=graph_view.graph,
            evidence=evidence,
            policy=self.policy or evolution_policy("fixed_artifact_patch"),
            memory_store=self.memory_store,
            skill_registry=self.skill_registry,
            permission_store=self.permission_store,
            artifact_state=self.artifact_state,
            skill_review_window=skill_review_window,
        )
        proposal: dict[str, Any] | None = None
        try:
            proposal = self.agent.propose(graph=graph_view.graph, evidence=evidence, workspace=workspace)
        except Exception as exc:
            deployment = EvolutionDeployment(
                parent_version=self.graph.version,
                next_version=next_version,
                task_id=case.public.task_id,
                candidate_id=None,
                status="planner_error",
                reason=f"{type(exc).__name__}: {exc}",
            )
            self._persist_deployment(deployment, proposal=proposal, workspace=workspace)
            self.graph = advance_graph_version(self.graph, version=next_version)
            self._write_json(self.root / "versions" / next_version / "graph.json", self.graph.to_dict())
            return deployment
        try:
            if proposal is None:
                raise ValueError("Evolution Agent returned no proposal")
            proposal = normalize_artifact_patch(proposal)
            if contains_source_identity(proposal):
                raise ValueError("Evolution proposal must not name a benchmark or source adapter")
            if self.skill_updates_locked and any(
                isinstance(update, dict) and str(update.get("path", "")).startswith("skills/")
                for update in proposal.get("file_updates", []) or []
            ):
                raise ValueError("Skill artifact updates are reserved for skill-consolidation rounds")
            validate_artifact_patch(proposal)
        except Exception as exc:
            deployment = EvolutionDeployment(
                parent_version=self.graph.version,
                next_version=next_version,
                task_id=case.public.task_id,
                candidate_id=None,
                status="rejected",
                reason=f"Artifact evolution policy rejected proposal: {exc}",
                validation={"accepted": False, "mode": "fixed_artifact_patch", "error_type": type(exc).__name__, "error": str(exc)},
            )
            self._persist_deployment(deployment, proposal=proposal, workspace=workspace)
            self.graph = advance_graph_version(self.graph, version=next_version)
            self._write_json(self.root / "versions" / next_version / "graph.json", self.graph.to_dict())
            return deployment

        try:
            updates = proposal.get("file_updates", [])
            runtime_updates: list[dict[str, Any]] = []
            if updates:
                episode_idx = _version_number(next_version, prefix="H")
                applied = apply_artifact_patch_to_runtime(
                    state=self.artifact_state or ArtifactState(prompt_addendum=""),
                    patch=proposal,
                    memory_store=self.memory_store,
                    permission_store=self.permission_store,
                    skill_registry=self.skill_registry,
                    next_version=f"A{episode_idx}",
                    next_skill_version=f"S{episode_idx}",
                    protect_existing_memory=self.protect_existing_memory,
                )
                self.artifact_state = applied["state"]
                runtime_updates = applied["runtime_updates"]
                if applied.get("skill_registry") is not None:
                    self.skill_registry = applied["skill_registry"]
        except Exception as exc:
            deployment = EvolutionDeployment(
                parent_version=self.graph.version,
                next_version=next_version,
                task_id=case.public.task_id,
                candidate_id=None,
                status="rejected",
                reason=f"Artifact candidate validation failed: {type(exc).__name__}: {exc}",
                validation={"accepted": False, "mode": "fixed_artifact_patch", "error_type": type(exc).__name__, "error": str(exc)},
            )
            self._persist_deployment(deployment, proposal=proposal, workspace=workspace)
            self.graph = advance_graph_version(self.graph, version=next_version)
            self._write_json(self.root / "versions" / next_version / "graph.json", self.graph.to_dict())
            return deployment

        self.graph = advance_graph_version(self.graph, version=next_version)
        status = "accepted" if updates else "no_change"
        deployment = EvolutionDeployment(
            parent_version=session.harness.config.version,
            next_version=next_version,
            task_id=case.public.task_id,
            candidate_id=None,
            status=status,
            reason=str(proposal.get("reason") or proposal.get("changes", [{}])[0].get("targeted_fix", "No artifact change is justified.")),
            validation={
                "accepted": bool(updates),
                "mode": "fixed_artifact_patch",
                "artifact_version": self.artifact_state.version if self.artifact_state is not None else None,
                "runtime_updates": runtime_updates,
                "evolution_workspace": str(workspace.root),
                "evolution_workspace_manifest": str(workspace.manifest_path),
            },
        )
        self._persist_deployment(deployment, proposal=proposal, workspace=workspace)
        self._write_json(self.root / "versions" / next_version / "graph.json", self.graph.to_dict())
        if status == "accepted" and self.artifact_state is not None:
            self._write_json(
                self.root / "artifact_versions" / self.artifact_state.version / "state.json",
                {
                    "schema_version": 1,
                    "version": self.artifact_state.version,
                    "prompt_addendum": self.artifact_state.prompt_addendum,
                    "guard_policy": self.artifact_state.guard_policy,
                },
            )
        return deployment

    def finalize_episode_storage(self, *, deployment: EvolutionDeployment, ordinal: int) -> Path | None:
        """Keep full evolution evidence only for an actual artifact update.

        Evolution and Guard triage need a complete workspace while an episode is
        running. A no-change or failed proposal has no new Harness state to
        preserve. Its lightweight
        decision remains in the append-only ledger and result row.
        """

        episode_dir = self.root / "episodes" / deployment.task_id
        if not episode_dir.exists():
            return None
        if deployment.status != "accepted":
            shutil.rmtree(episode_dir)
            return None
        retained = self.root / "updates" / f"{ordinal:04d}_{deployment.task_id}"
        retained.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if retained.exists():
            raise FileExistsError(f"Retained evolution update already exists: {retained}")
        shutil.move(str(episode_dir), str(retained))
        if isinstance(deployment.validation, dict):
            deployment.validation["evolution_workspace"] = str(retained / "evolution_workspace")
            deployment.validation["evolution_workspace_manifest"] = str(retained / "evolution_workspace" / "workspace_manifest.json")
            deployment.validation["retained_update_ordinal"] = ordinal
        # Store the deployment with its final retained workspace paths.
        self._write_json(retained / "deployment.json", deployment.to_dict())
        self._write_json(
            retained / "workspace_reference.json",
            {
                "workspace": str(retained / "evolution_workspace"),
                "manifest": str(retained / "evolution_workspace" / "workspace_manifest.json"),
                "full_public_trace": str(retained / "evolution_workspace" / "episode" / "public_trace.json"),
                "private_evaluator_included": False,
            },
        )
        return retained

    def evolve_batch(
        self,
        *,
        episodes: list[BatchEvolutionEpisode],
        next_version: str,
    ) -> EvolutionDeployment:
        """Ask the agent for one artifact patch after a completed episode batch."""

        if not episodes:
            raise ValueError("Batch evolution requires at least one completed episode")
        graph_view = source_blind_graph_view(self.graph)
        episode_evidence = [
            build_online_evidence(
                case=episode.case,
                session=episode.session,
                feedback=episode.feedback,
                trace_records=episode.trace_records,
                processor_id_aliases=graph_view.actual_to_view_processor_id,
            )
            for episode in episodes
        ]
        evidence = build_batch_online_evidence(
            episode_evidence=episode_evidence,
            batch_id=_batch_task_id(episodes),
            next_version=next_version,
        )
        skill_review_window = max(self.skill_evolution_interval, MIN_SKILL_CREATE_EPISODES)
        workspace = create_evolution_workspace(
            evolution_root=self.root,
            task_id=evidence["batch"]["batch_id"],
            graph=graph_view.graph,
            evidence=evidence,
            policy=self.policy or evolution_policy("fixed_artifact_patch"),
            memory_store=self.memory_store,
            skill_registry=self.skill_registry,
            permission_store=self.permission_store,
            artifact_state=self.artifact_state,
            skill_review_window=skill_review_window,
        )
        proposal: dict[str, Any] | None = None
        parent_version = self.graph.version
        try:
            proposal = self.agent.propose(graph=graph_view.graph, evidence=evidence, workspace=workspace)
        except Exception as exc:
            deployment = EvolutionDeployment(
                parent_version=parent_version,
                next_version=next_version,
                task_id=evidence["batch"]["batch_id"],
                candidate_id=None,
                status="planner_error",
                reason=f"{type(exc).__name__}: {exc}",
            )
            self._persist_deployment(deployment, proposal=proposal, workspace=workspace)
            self.graph = advance_graph_version(self.graph, version=next_version)
            self._write_json(self.root / "versions" / next_version / "graph.json", self.graph.to_dict())
            return deployment
        try:
            if proposal is None:
                raise ValueError("Evolution Agent returned no proposal")
            proposal = normalize_artifact_patch(proposal)
            if contains_source_identity(proposal):
                raise ValueError("Evolution proposal must not name a benchmark or source adapter")
            validate_artifact_patch(proposal)
        except Exception as exc:
            deployment = EvolutionDeployment(
                parent_version=parent_version,
                next_version=next_version,
                task_id=evidence["batch"]["batch_id"],
                candidate_id=None,
                status="rejected",
                reason=f"Artifact evolution policy rejected proposal: {exc}",
                validation={"accepted": False, "mode": "fixed_artifact_patch", "error_type": type(exc).__name__, "error": str(exc)},
            )
            self._persist_deployment(deployment, proposal=proposal, workspace=workspace)
            self.graph = advance_graph_version(self.graph, version=next_version)
            self._write_json(self.root / "versions" / next_version / "graph.json", self.graph.to_dict())
            return deployment
        try:
            updates = proposal.get("file_updates", [])
            runtime_updates: list[dict[str, Any]] = []
            if updates:
                episode_idx = _version_number(next_version, prefix="H")
                applied = apply_artifact_patch_to_runtime(
                    state=self.artifact_state or ArtifactState(prompt_addendum=""),
                    patch=proposal,
                    memory_store=self.memory_store,
                    permission_store=self.permission_store,
                    skill_registry=self.skill_registry,
                    next_version=f"A{episode_idx}",
                    next_skill_version=f"S{episode_idx}",
                    protect_existing_memory=self.protect_existing_memory,
                )
                self.artifact_state = applied["state"]
                runtime_updates = applied["runtime_updates"]
                if applied.get("skill_registry") is not None:
                    self.skill_registry = applied["skill_registry"]
        except Exception as exc:
            deployment = EvolutionDeployment(
                parent_version=parent_version,
                next_version=next_version,
                task_id=evidence["batch"]["batch_id"],
                candidate_id=None,
                status="rejected",
                reason=f"Artifact candidate validation failed: {type(exc).__name__}: {exc}",
                validation={"accepted": False, "mode": "fixed_artifact_patch", "error_type": type(exc).__name__, "error": str(exc)},
            )
            self._persist_deployment(deployment, proposal=proposal, workspace=workspace)
            self.graph = advance_graph_version(self.graph, version=next_version)
            self._write_json(self.root / "versions" / next_version / "graph.json", self.graph.to_dict())
            return deployment

        self.graph = advance_graph_version(self.graph, version=next_version)
        status = "accepted" if updates else "no_change"
        deployment = EvolutionDeployment(
            parent_version=parent_version,
            next_version=next_version,
            task_id=evidence["batch"]["batch_id"],
            candidate_id=None,
            status=status,
            reason=str(proposal.get("reason") or proposal.get("changes", [{}])[0].get("targeted_fix", "No artifact change is justified.")),
            validation={
                "accepted": bool(updates),
                "mode": "fixed_artifact_patch",
                "evolution_granularity": "batch",
                "batch_size": len(episodes),
                "batch_task_ids": [episode.case.public.task_id for episode in episodes],
                "artifact_version": self.artifact_state.version if self.artifact_state is not None else None,
                "runtime_updates": runtime_updates,
                "evolution_workspace": str(workspace.root),
                "evolution_workspace_manifest": str(workspace.manifest_path),
            },
        )
        self._persist_deployment(deployment, proposal=proposal, workspace=workspace)
        self._write_json(self.root / "versions" / next_version / "graph.json", self.graph.to_dict())
        if status == "accepted" and self.artifact_state is not None:
            self._write_json(
                self.root / "artifact_versions" / self.artifact_state.version / "state.json",
                {
                    "schema_version": 1,
                    "version": self.artifact_state.version,
                    "prompt_addendum": self.artifact_state.prompt_addendum,
                    "guard_policy": self.artifact_state.guard_policy,
                },
            )
        return deployment

    def _maybe_deploy_skill_candidate(
        self,
        *,
        workspace: EvolutionWorkspace,
        evidence_trace_hashes: tuple[str, ...],
        next_graph_version: str,
    ) -> dict[str, Any]:
        """Validate and atomically publish the candidate Skill set for next episode.

        Skill authoring uses file-based assets. The controller
        never invents a Skill: it accepts only a complete ``SKILL.md`` plus an
        explicit manifest written by the Evolution Agent in this workspace.
        """

        registry = self.skill_registry
        if registry is None:
            return {"status": "disabled", "reason": "skill_runtime_disabled"}
        manifest = workspace.candidate_root / "skills" / "manifest.json"
        if not manifest.is_file():
            return {"status": "no_candidate", "registry_version": registry.version}
        episode_number = _version_number(next_graph_version, prefix="H")
        if episode_number % self.skill_evolution_interval != 0:
            return {
                "status": "deferred_not_due",
                "registry_version": registry.version,
                "interval": self.skill_evolution_interval,
                "episode_number": episode_number,
            }
        try:
            review = EvolutionWorkspaceTools(
                workspace=workspace,
                graph=self.graph,
                policy=self.policy or evolution_policy("fixed_artifact_patch"),
                evidence={},
                evidence_trace_hashes=evidence_trace_hashes,
            ).require_source_review_for_skill_manifest(manifest)
            successor = load_skill_manifest(
                manifest,
                parent=registry,
                next_version=f"S{_version_number(registry.version, prefix='S') + 1}",
                evidence_trace_hashes=evidence_trace_hashes,
                forbidden_literals=_workspace_task_specific_literals(workspace),
            )
        except Exception as exc:
            return {"status": "rejected", "error_type": type(exc).__name__, "error": str(exc)}
        successor.write(self.root.parent / "skills" / "versions" / successor.version / "registry.json")
        self.skill_registry = successor
        return {
            "status": "accepted",
            "parent_version": registry.version,
            "next_version": successor.version,
            "parent_hash": registry.registry_hash,
            "next_hash": successor.registry_hash,
            "active_skill_count": len(successor.active),
            "source_review": review,
        }

    def _persist_deployment(
        self,
        deployment: EvolutionDeployment,
        *,
        proposal: dict[str, Any] | None,
        workspace: EvolutionWorkspace,
    ) -> None:
        directory = self.root / "episodes" / deployment.task_id
        self._write_json(directory / "deployment.json", deployment.to_dict())
        if proposal is not None:
            self._write_json(directory / "proposal.json", proposal)
        self._write_json(
            directory / "workspace_reference.json",
            {
                "workspace": str(workspace.root),
                "manifest": str(workspace.manifest_path),
                "full_public_trace": str(workspace.root / "episode" / "public_trace.json"),
                "private_evaluator_included": False,
            },
        )
        agent_session = getattr(self.agent, "last_session", None)
        if isinstance(agent_session, list):
            self._write_json(directory / "evolution_agent_session.json", agent_session)
        tool_audit = getattr(self.agent, "last_tool_audit", None)
        if isinstance(tool_audit, list):
            source_review_path = workspace.root / "harness" / "source_review_status.json"
            source_review: dict[str, Any] | None = None
            if source_review_path.is_file():
                value = json.loads(source_review_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    source_review = value
            self._write_json(
                directory / "evolution_tool_audit.json",
                {
                    "schema_version": 1,
                    "task_id": deployment.task_id,
                    "tool_call_count": len(tool_audit),
                    "tool_calls": copy.deepcopy(tool_audit),
                    "session_artifact": "evolution_agent_session.json",
                    "source_review_status": source_review,
                    "recording_contract": (
                        "Each tool_calls entry retains the complete tool message returned to the model, "
                        "its SHA-256 and byte count, the request arguments, and source-review state after the call."
                    ),
                },
            )
        request_audit = getattr(self.agent, "last_request_audit", None)
        if isinstance(request_audit, list):
            self._write_json(
                directory / "evolution_request_audit.json",
                {
                    "schema_version": 1,
                    "task_id": deployment.task_id,
                    "request_count": len(request_audit),
                    "requests": copy.deepcopy(request_audit),
                    "recording_contract": (
                        "Each request entry records whether an Evolution Agent model call used the primary "
                        "or fallback endpoint and whether the attempt succeeded before any workspace tool effect."
                    ),
                },
            )
        self._append_ledger(deployment, proposal=proposal)

    def _append_ledger(self, deployment: EvolutionDeployment, *, proposal: dict[str, Any] | None = None) -> None:
        """Keep prompt-free, source-blind deployment and artifact-patch history."""

        path = self.root / "ledger" / "deployments.jsonl"
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(deployment.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        path.chmod(0o600)
        patch_path = self.root / "ledger" / "artifact_patch_ledger.jsonl"
        patch_row = _source_blind_patch_ledger_row(deployment=deployment, proposal=proposal)
        with patch_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(patch_row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        patch_path.chmod(0o600)

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        path.chmod(0o600)



def _source_blind_patch_ledger_row(*, deployment: EvolutionDeployment, proposal: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize prior artifact decisions without benchmark/source provenance."""

    validation = deployment.validation or {}
    changes = proposal.get("changes", []) if isinstance(proposal, dict) else []
    updates = proposal.get("file_updates", []) if isinstance(proposal, dict) else []
    return {
        "schema_version": 1,
        "task_id": deployment.task_id,
        "parent_version": deployment.parent_version,
        "next_version": deployment.next_version,
        "status": deployment.status,
        "reason": deployment.reason,
        "artifact_version": validation.get("artifact_version") if isinstance(validation, dict) else None,
        "changes": _source_blind_changes(changes),
        "file_updates": _source_blind_file_updates(updates),
        "runtime_updates": copy.deepcopy(validation.get("runtime_updates", [])) if isinstance(validation, dict) else [],
    }


def _source_blind_changes(changes: Any) -> list[dict[str, Any]]:
    if not isinstance(changes, list):
        return []
    allowed = {
        "change_id",
        "change_type",
        "component",
        "files",
        "failure_pattern",
        "evidence_cases",
        "failure_evidence",
        "root_cause",
        "targeted_fix",
        "change_summary",
        "predicted_fixes",
        "risk_cases",
        "constraint_level",
        "why_this_component",
        "acceptance_rule",
    }
    out = []
    for change in changes:
        if isinstance(change, dict):
            out.append({key: copy.deepcopy(value) for key, value in change.items() if key in allowed})
    return out


def _source_blind_file_updates(updates: Any) -> list[dict[str, Any]]:
    if not isinstance(updates, list):
        return []
    out = []
    for update in updates:
        if not isinstance(update, dict):
            continue
        item = {
            "path": update.get("path"),
            "mode": update.get("mode"),
        }
        if isinstance(update.get("records"), list):
            item["record_count"] = len(update["records"])
        if isinstance(update.get("operations"), list):
            item["operation_count"] = len(update["operations"])
            item["operation_paths"] = [op.get("path") for op in update["operations"] if isinstance(op, dict)]
        if isinstance(update.get("text"), str):
            item["text_bytes"] = len(update["text"].encode("utf-8"))
        out.append(item)
    return out


def build_online_evidence(
    *,
    case: SafeCoEvoCase,
    session: HarnessSession,
    feedback: OfficialEpisodeFeedback,
    trace_records: list[dict[str, Any]],
    processor_id_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the post-episode evidence exposed to the Evolution Agent.

    The evidence has public task/trajectory data and released, normalized
    official feedback. It deliberately excludes benchmark provenance, native
    oracle schemas, private evaluator records, and any opportunity to modify
    the original trace.
    """

    public_trace = source_blind_trace(
        _full_public_trace(trace_records),
        processor_id_aliases=processor_id_aliases,
    )
    return {
        "schema_version": 2,
        "view_contract": "source_blind_evolution_view",
        "timing": "after_episode_before_next_episode",
        "task": session.task.agent_payload(),
        "official_feedback": source_blind_feedback(feedback),
        "evolution_signals": copy.deepcopy(session.state.evolution_signals),
        "evolution_evidence": copy.deepcopy(session.state.evolution_evidence),
        "trace_hashes": list(session.state.annotations.get("trace_hashes", ())),
        "public_trace": public_trace,
        "public_trace_delivery": "complete_episode_trajectory_direct",
        "raw_trace_integrity": "The complete source-blind episode trajectory is supplied here as a final conversation plus action timeline; raw append-only snapshot records remain on disk, bound by their trace hash chain.",
        "private_evaluator_included": False,
        "benchmark_provenance_included": False,
        "completed_episode_mutable": False,
    }


def build_batch_online_evidence(
    *,
    episode_evidence: list[dict[str, Any]],
    batch_id: str,
    next_version: str,
) -> dict[str, Any]:
    """Build one source-blind evidence package for batch-level evolution."""

    safety_counts: dict[str, int] = {}
    goal_counts: dict[str, int] = {}
    trace_hashes: list[str] = []
    for evidence in episode_evidence:
        feedback = evidence.get("official_feedback", {})
        outcomes = feedback.get("outcomes", {}) if isinstance(feedback, dict) else {}
        for bucket, key in ((safety_counts, "safety"), (goal_counts, "goal")):
            value = outcomes.get(key) if isinstance(outcomes, dict) else None
            bucket[str(value or "unknown")] = bucket.get(str(value or "unknown"), 0) + 1
        trace_hashes.extend(str(item) for item in evidence.get("trace_hashes", ()) if item)
    return {
        "schema_version": 2,
        "view_contract": "source_blind_batch_evolution_view",
        "timing": "after_batch_before_next_episode",
        "task": {
            "task_id": batch_id,
            "batch": True,
            "episode_count": len(episode_evidence),
            "episode_task_ids": [
                str((evidence.get("task") or {}).get("task_id", f"episode_{index}"))
                for index, evidence in enumerate(episode_evidence, start=1)
                if isinstance(evidence.get("task"), dict)
            ],
        },
        "batch": {
            "batch_id": batch_id,
            "episode_count": len(episode_evidence),
            "next_harness_version": next_version,
            "evolution_granularity": "batch",
            "deployment_timing": "after_all_batch_episodes_before_next_episode",
        },
        "official_feedback": {
            "schema_version": 1,
            "batch_summary": {
                "safety": dict(sorted(safety_counts.items())),
                "goal": dict(sorted(goal_counts.items())),
            },
            "episodes": [copy.deepcopy(evidence.get("official_feedback", {})) for evidence in episode_evidence],
            "feedback_contract": "normalized_official_outcomes_without_benchmark_or_oracle_identity",
        },
        "episodes": [
            {
                **{key: copy.deepcopy(value) for key, value in evidence.items() if key != "public_trace"},
                "public_trace": {
                    "delivery": "see_top_level_public_trace",
                    "episode_index": index,
                },
            }
            for index, evidence in enumerate(episode_evidence, start=1)
        ],
        "trace_hashes": trace_hashes,
        "public_trace": [
            {
                "episode_index": index,
                "task_id": (evidence.get("task") or {}).get("task_id"),
                "trace": copy.deepcopy(evidence.get("public_trace", [])),
            }
            for index, evidence in enumerate(episode_evidence, start=1)
        ],
        "public_trace_delivery": "complete_episode_trajectory_direct_batch",
        "raw_trace_integrity": "The complete source-blind episode trajectories for every batch episode are supplied here as final conversations plus action timelines; raw append-only snapshot records remain on disk.",
        "private_evaluator_included": False,
        "benchmark_provenance_included": False,
        "completed_episode_mutable": False,
    }


def _batch_task_id(episodes: list[BatchEvolutionEpisode]) -> str:
    first = episodes[0].case.public.task_id
    last = episodes[-1].case.public.task_id
    digest = json.dumps([episode.case.public.task_id for episode in episodes], sort_keys=True).encode("utf-8")
    return f"batch_{len(episodes):03d}_{first[:8]}_{last[:8]}_{hashlib.sha256(digest).hexdigest()[:10]}"
