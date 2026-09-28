"""Composable SafeCoEvo runtime for online safety evaluation and evolution."""

from .contracts import CandidateAction, GuardVerdict, Hook, PermissionExperienceItem, TaskOutcome, ToolCall
from .artifact_patcher import ArtifactState, apply_artifact_patch_to_runtime, hydrate_seed_memory_store, load_seed_artifact_state, validate_artifact_patch
from .dataset import SafeCoEvoCase, load_safecoevo_dataset
from .execution import AgentDoGGuardAdapter, LiveAgentSecurityBenchExecutor, LiveAgentSafetyBenchExecutor, NativeExecution
from .runtime import ActionExecution, HarnessBuilder, HarnessSession, SafetyHarness, build_minimal_harness
from .permission import InMemoryPermissionExperienceStore, PermissionExperienceStore
from .skills import SkillArtifact, SkillMatch, SkillRegistry, format_skill_reference, load_seed_skill_registry, load_skill_manifest
from .online_feedback import HarnessDelta, OfficialEpisodeFeedback, OnlineEvolutionState, official_episode_feedback
from .evolution_graph import (
    EvolutionDeployment,
    EvolutionPolicy,
    OpenAICompatibleEvolutionAgent,
    ProcessorGraph,
    ProcessorNode,
    StaticEvolutionAgent,
    advance_graph_version,
    build_harness_from_graph,
    evolution_policy,
    initial_processor_graph,
    processor_graph_from_dict,
)
from .online_evolution import BatchEvolutionEpisode, OnlineGraphEvolutionController
from .evolution_workspace import EvolutionWorkspace, EvolutionWorkspaceTools, create_evolution_workspace

__all__ = [
    "ArtifactState",
    "apply_artifact_patch_to_runtime",
    "hydrate_seed_memory_store",
    "load_seed_artifact_state",
    "validate_artifact_patch",
    "CandidateAction",
    "AgentDoGGuardAdapter",
    "GuardVerdict",
    "PermissionExperienceItem",
    "ActionExecution",
    "HarnessBuilder",
    "HarnessSession",
    "InMemoryPermissionExperienceStore",
    "PermissionExperienceStore",
    "SkillArtifact",
    "SkillMatch",
    "SkillRegistry",
    "format_skill_reference",
    "load_seed_skill_registry",
    "load_skill_manifest",
    "Hook",
    "LiveAgentSecurityBenchExecutor",
    "LiveAgentSafetyBenchExecutor",
    "NativeExecution",
    "OfficialEpisodeFeedback",
    "HarnessDelta",
    "OnlineEvolutionState",
    "official_episode_feedback",
    "SafeCoEvoCase",
    "SafetyHarness",
    "TaskOutcome",
    "ToolCall",
    "build_minimal_harness",
    "load_safecoevo_dataset",
    "ProcessorNode",
    "ProcessorGraph",
    "EvolutionDeployment",
    "EvolutionPolicy",
    "StaticEvolutionAgent",
    "OpenAICompatibleEvolutionAgent",
    "initial_processor_graph",
    "build_harness_from_graph",
    "evolution_policy",
    "advance_graph_version",
    "processor_graph_from_dict",
    "OnlineGraphEvolutionController",
    "BatchEvolutionEpisode",
    "EvolutionWorkspace",
    "EvolutionWorkspaceTools",
    "create_evolution_workspace",
]
