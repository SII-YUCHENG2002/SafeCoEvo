"""Initial atomic Processors for the evolvable Safety Harness."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .contracts import CandidateAction, GuardVerdict, Hook, Resource, ToolCall
from .events import HarnessEvent, agent_visible_messages
from .memory import MemoryStore
from .permission import PermissionExperienceStore
from .processor import MultiHookProcessor, Processor, ProcessorSpec
from .skills import SkillRegistry, format_skill_reference
from .target_prompts import (
    format_target_artifact_context,
    format_guard_preflight_result,
    online_target_system_message,
)
from .trace import TraceStore, event_snapshot


@dataclass(frozen=True)
class GuardRequest:
    """Only public agent-visible state is supplied to a Guard implementation."""

    task_id: str
    hook: Hook
    messages: tuple[dict, ...]
    available_tools: tuple[dict, ...]
    action: object
    # Runtime boundary being diagnosed; tool actions are the default.
    subject_type: str = "tool_action"
    source: str = "runtime"
    latest_event: dict[str, Any] | None = None
    trusted_harness_context: dict[str, Any] | None = None


class Guard(Protocol):
    def diagnose(self, request: GuardRequest) -> GuardVerdict: ...


class CallableGuard:
    """Adapter for tests or local integrations that expose a plain callable."""

    def __init__(self, handler: Callable[[GuardRequest], GuardVerdict]) -> None:
        self._handler = handler

    def diagnose(self, request: GuardRequest) -> GuardVerdict:
        return self._handler(request)


def _guard_event_digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _message_guard_event_id(*, event_type: str, message_index: int, content: object) -> str:
    return f"{event_type}:{message_index}:{_guard_event_digest(content)}"


def _tool_action_guard_event(action: CandidateAction, *, message_index: int) -> dict[str, Any]:
    if action.tool_call is None:
        raise ValueError("A tool-action Guard event requires a tool call")
    payload = {
        "tool_call_id": action.tool_call.call_id,
        "tool_name": action.tool_call.name,
        "arguments": copy.deepcopy(action.tool_call.arguments),
        "execution_state": "proposed_not_executed",
    }
    return {
        "event_id": f"tool_action:{message_index}:{action.tool_call.call_id}:{_guard_event_digest(payload)}",
        "event_type": "tool_action",
        "source": "native_tool_proposal",
        **payload,
    }


def _claim_guard_event(event: HarnessEvent, event_id: str) -> bool:
    inspected = event.state.annotations.setdefault("guard_inspected_event_ids", [])
    if event_id in inspected:
        return False
    inspected.append(event_id)
    return True


def target_artifact_context_for_event(event: HarnessEvent) -> str:
    """Build the Target-visible artifact block from current per-task state."""

    guard_runtime = dict(event.state.annotations.get("guard_runtime") or {})
    permission_lessons = event.state.annotations.get("permission_experience")
    skill_references = event.state.annotations.get("skill_references")
    guard_observations = event.state.annotations.get("guard_observation_summaries")
    return format_target_artifact_context(
        processor_names=event.state.annotations.get("active_processors"),
        skill_runtime=event.state.annotations.get("skill_runtime"),
        skill_runtime_snapshot=event.state.annotations.get("skill_runtime_snapshot"),
        skill_catalog=event.state.annotations.get("skill_catalog"),
        skill_references=skill_references if isinstance(skill_references, list) else [],
        memory_runtime=event.state.annotations.get("memory_runtime"),
        memory_lessons=[item.content for item in event.state.retrieved_memory],
        permission_runtime=event.state.annotations.get("permission_runtime"),
        permission_lessons=permission_lessons if isinstance(permission_lessons, list) else [],
        guard_runtime=guard_runtime,
        guard_policy=event.state.annotations.get("guard_policy"),
        guard_observations=guard_observations if isinstance(guard_observations, list) else [],
    )


def refresh_target_system_message(event: HarnessEvent) -> None:
    """Refresh the single Target system message after artifact state changes."""

    message = online_target_system_message(
        base_system=event.task.system,
        artifact_context=target_artifact_context_for_event(event),
        addendum_template=event.state.annotations.get("prompt_addendum_template"),
    )
    if event.state.formatted_messages:
        event.state.formatted_messages[0] = copy.deepcopy(message)
    else:
        event.state.formatted_messages.append(copy.deepcopy(message))
    if event.model_messages:
        event.model_messages[0] = copy.deepcopy(message)


def append_guard_observation_summary(
    event: HarnessEvent,
    *,
    action_id: str,
    tool_name: str,
    verdict: GuardVerdict,
    kind: str,
) -> None:
    """Expose prior Guard results in the Guard section of the next system prompt."""

    summary = (
        f"{kind}: action_id={action_id}; tool={tool_name}; safety={verdict.safety}; "
        f"reason={verdict.reason or 'none'}"
    )
    event.state.annotations.setdefault("guard_observation_summaries", []).append(summary)
    refresh_target_system_message(event)


class TraceRecorder(MultiHookProcessor):
    """Captures post-processor snapshots in an append-only hash chain."""

    def __init__(self, store: TraceStore) -> None:
        self.store = store
        self.spec = ProcessorSpec(
            name="TraceRecorder",
            hooks=frozenset(Hook),
            order=1000,
            reads=frozenset({Resource.CONTEXT, Resource.ACTION, Resource.TOOL, Resource.TRACE}),
            writes=frozenset({Resource.TRACE}),
        )

    def process(self, event: HarnessEvent) -> None:
        trace_hash = self.store.append(event_snapshot(event))
        event.state.annotations.setdefault("trace_hashes", []).append(trace_hash)


class PromptFormatter(Processor):
    """Builds online Target input from the current Prompt and Guard artifacts."""

    def __init__(self, addendum_template: str | None = None, guard_policy: dict[str, Any] | None = None) -> None:
        self.addendum_template = addendum_template
        self.guard_policy = copy.deepcopy(guard_policy or {})
        self.spec = ProcessorSpec(
            name="PromptFormatter",
            hooks=frozenset({Hook.TASK_START}),
            order=30,
            reads=frozenset({Resource.PROMPT, Resource.CONTEXT, Resource.MEMORY}),
            writes=frozenset({Resource.PROMPT}),
        )

    def process(self, event: HarnessEvent) -> None:
        if self.addendum_template:
            event.state.annotations["prompt_addendum_template"] = self.addendum_template
        if self.guard_policy:
            event.state.annotations["guard_policy"] = copy.deepcopy(self.guard_policy)
        guard_runtime = dict(event.state.annotations.get("guard_runtime") or {})
        if guard_runtime.get("enabled"):
            existing_names = {
                str(tool.get("function", {}).get("name", ""))
                for tool in event.task.tools
                if isinstance(tool, dict)
            }
            skill_tool = event.state.annotations.get("skill_internal_tool_name")
            if isinstance(skill_tool, str):
                existing_names.add(skill_tool)
            preflight_name = "CheckGuard"
            suffix = 1
            while preflight_name in existing_names:
                suffix += 1
                preflight_name = f"CheckGuard{suffix}"
            guard_runtime["preflight_tool_name"] = preflight_name
            event.state.annotations["guard_runtime"] = guard_runtime
        messages = [
            online_target_system_message(
                base_system=event.task.system,
                artifact_context=target_artifact_context_for_event(event),
                addendum_template=event.state.annotations.get("prompt_addendum_template"),
            ),
            *copy.deepcopy(event.state.messages),
        ]
        event.state.formatted_messages = messages


class SkillCatalogProcessor(Processor):
    """Expose only a compact active-Skill directory at task start."""

    def __init__(self, registry: SkillRegistry, *, enabled: bool = True, max_active: int = 8) -> None:
        if max_active < 1:
            raise ValueError("Skill catalog max_active must be at least 1")
        self.registry = registry
        self.enabled = enabled
        self.max_active = max_active
        self.spec = ProcessorSpec(
            name="SkillCatalogProcessor",
            hooks=frozenset({Hook.TASK_START}),
            order=25,
            reads=frozenset({Resource.SKILL, Resource.CONTEXT}),
            writes=frozenset({Resource.SKILL, Resource.PROMPT}),
        )

    def process(self, event: HarnessEvent) -> None:
        if not self.enabled:
            event.state.annotations["skill_runtime"] = {"enabled": False}
            return
        catalog = self.registry.catalog(limit=self.max_active)
        names = {
            str(tool.get("function", {}).get("name", ""))
            for tool in event.task.tools
            if isinstance(tool, dict)
        }
        internal_name = "LoadSkill"
        suffix = 1
        while internal_name in names:
            suffix += 1
            internal_name = f"LoadSkill{suffix}"
        lines = ["<available_skills>"]
        for item in catalog:
            lines.append(
                f"- name={item['name']} | id={item['skill_id']} | version={item['version']} | "
                f"capability={item['capability']} | description={item['description']}"
            )
        lines.append("</available_skills>")
        lines.append(
            f"The listed Skills are optional advisory procedures. You may use the Harness `{internal_name}` tool "
            "with a listed name when you need its complete reference; the original task and tool schemas remain authoritative."
        )
        event.state.annotations.update(
            {
                "skill_runtime": {
                    "enabled": True,
                    "registry_version": self.registry.version,
                    "registry_hash": self.registry.registry_hash,
                    "skills_advertised": catalog,
                    "internal_tool_name": internal_name,
                },
                "skill_catalog": "\n".join(lines),
                "skill_internal_tool_name": internal_name,
                "skill_usage": [],
            }
        )


class PermissionExperienceInjector(Processor):
    """Inject high-level permission lessons without changing agent decisions."""

    def __init__(self, store: PermissionExperienceStore, *, enabled: bool = True, limit: int = 3) -> None:
        if limit < 1:
            raise ValueError("Permission experience limit must be positive")
        self.store = store
        self.enabled = enabled
        self.limit = limit
        self.spec = ProcessorSpec(
            name="PermissionExperienceInjector",
            hooks=frozenset({Hook.TASK_START}),
            order=27,
            reads=frozenset({Resource.CONTEXT, Resource.PERMISSION}),
            writes=frozenset({Resource.PERMISSION, Resource.PROMPT}),
            after=frozenset({"SkillRuntimeProcessor"}),
        )

    def process(self, event: HarnessEvent) -> None:
        event.state.annotations["permission_runtime"] = {
            "enabled": self.enabled,
            "limit": self.limit,
            "resource": "harness://permission/high_level_experience",
        }
        if not self.enabled:
            event.state.annotations["permission_experience"] = []
            return
        tool_names = " ".join(
            str(tool.get("function", {}).get("name", ""))
            for tool in event.task.tools
            if isinstance(tool, dict)
        )
        query = "\n".join(
            [
                *(str(message.get("content", "")) for message in event.task.messages),
                tool_names,
            ]
        )
        items = self.store.retrieve(query, limit=self.limit)
        event.state.annotations["permission_experience"] = [item.content for item in items]
        event.state.annotations["permission_runtime"]["returned_count"] = len(items)


class SkillRuntimeProcessor(Processor):
    """Record the active Registry snapshot at a task boundary.

    The registry object is bound when a new Harness session is built, so a
    deployed version can only affect a subsequent episode.  This processor is
    still explicit in the graph to make the version transition traceable.
    """

    def __init__(self, registry: SkillRegistry, *, enabled: bool = True) -> None:
        self.registry = registry
        self.enabled = enabled
        self.spec = ProcessorSpec(
            name="SkillRuntimeProcessor",
            hooks=frozenset({Hook.TASK_START}),
            order=26,
            reads=frozenset({Resource.SKILL}),
            writes=frozenset({Resource.SKILL, Resource.TRACE}),
            after=frozenset({"skill_catalog"}),
        )

    def process(self, event: HarnessEvent) -> None:
        if not self.enabled:
            return
        event.state.annotations["skill_runtime_snapshot"] = {
            "registry_version": self.registry.version,
            "registry_hash": self.registry.registry_hash,
            "active_skill_count": len(self.registry.active),
            "retrieval_mode": self.registry.retrieval_mode,
        }


class ProgressiveSkillLoader(Processor):
    """Inject top-matching Skill references into only the first model request."""

    def __init__(self, registry: SkillRegistry, *, enabled: bool = True, top_k: int = 3) -> None:
        self.registry = registry
        self.enabled = enabled
        self.top_k = top_k
        self.spec = ProcessorSpec(
            name="ProgressiveSkillLoader",
            hooks=frozenset({Hook.BEFORE_MODEL}),
            order=30,
            reads=frozenset({Resource.SKILL, Resource.CONTEXT, Resource.PROMPT}),
            writes=frozenset({Resource.SKILL, Resource.PROMPT, Resource.TRACE}),
        )

    def process(self, event: HarnessEvent) -> None:
        if not self.enabled or event.state.annotations.get("skill_auto_injection_attempted"):
            return
        event.state.annotations["skill_auto_injection_attempted"] = True
        query = "\n".join(
            [
                *(str(message.get("content", "")) for message in event.task.messages),
                *(
                    str(tool.get("function", {}).get("name", ""))
                    for tool in event.task.tools
                    if isinstance(tool, dict)
                ),
            ]
        )
        matches = self.registry.retrieve(query, limit=self.top_k)
        event.state.annotations["skill_retrieval"] = self.registry.retrieval_metadata()
        if not matches:
            return
        if event.model_messages is None:
            raise RuntimeError("ProgressiveSkillLoader requires before_model request messages")
        references: list[str] = []
        usage: list[dict[str, Any]] = event.state.annotations.setdefault("skill_usage", [])
        for match in matches:
            skill = self.registry.get(match.skill_id)
            if skill is None:  # Registry changed outside a bound session.
                continue
            references.append(format_skill_reference(skill))
            usage.append(
                {
                    "kind": "auto_injected",
                    "request_sequence": event.sequence,
                    "match": match.to_dict(),
                    "content_bytes": len(skill.content.encode("utf-8")),
                }
            )
        if references:
            event.state.annotations["skill_references"] = list(references)
            refresh_target_system_message(event)


class SkillToolProvider(Processor):
    """Resolve Harness-internal ``LoadSkill`` calls without a native executor."""

    def __init__(self, registry: SkillRegistry, *, enabled: bool = True, expose_tool: bool = True) -> None:
        self.registry = registry
        self.enabled = enabled
        self.expose_tool = expose_tool
        self.spec = ProcessorSpec(
            name="SkillToolProvider",
            hooks=frozenset({Hook.BEFORE_MODEL, Hook.BEFORE_TOOL}),
            order=40,
            reads=frozenset({Resource.SKILL, Resource.TOOL, Resource.ACTION}),
            writes=frozenset({Resource.SKILL, Resource.TOOL, Resource.TRACE}),
        )

    def process(self, event: HarnessEvent) -> HarnessEvent | None:
        if not self.enabled:
            return None
        internal_name = str(event.state.annotations.get("skill_internal_tool_name", "LoadSkill"))
        if event.hook == Hook.BEFORE_MODEL:
            if not self.expose_tool or event.available_tools is None:
                return None
            if not event.task.tools:
                # Sources without callable tools can still receive automatic
                # references but should not see an unusable function schema.
                return None
            event.available_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": internal_name,
                        "description": "Read the full text of one listed Harness Skill by name or id. This is a read-only Harness-internal reference.",
                        "parameters": {
                            "type": "object",
                            "properties": {"name": {"type": "string", "description": "A name or id from <available_skills>."}},
                            "required": ["name"],
                            "additionalProperties": False,
                        },
                },
                }
            )
            return None
        proposed = event.proposed_action or event.action
        if proposed is None or proposed.kind != "tool" or proposed.tool_call is None or proposed.tool_call.name != internal_name:
            return None
        requested = proposed.tool_call.arguments.get("name")
        skill = self.registry.get(str(requested)) if isinstance(requested, str) else None
        usage: list[dict[str, Any]] = event.state.annotations.setdefault("skill_usage", [])
        if skill is None:
            usage.append({"kind": "load_miss", "request_sequence": event.sequence, "requested": str(requested)[:128]})
            result = (
                "[Harness Skill lookup]\nNo active Skill matches that name. "
                "Choose only a name or id listed in <available_skills>, or continue using the original task tools."
            )
        else:
            usage.append(
                {
                    "kind": "loaded_by_agent",
                    "request_sequence": event.sequence,
                    "skill_id": skill.skill_id,
                    "name": skill.name,
                    "content_sha256": skill.content_sha256,
                    "content_bytes": len(skill.content.encode("utf-8")),
                }
            )
            result = format_skill_reference(skill)
            refs: list[str] = event.state.annotations.setdefault("skill_references", [])
            if result not in refs:
                refs.append(result)
            refresh_target_system_message(event)
        return event.with_internal_tool_result(result=result, processor_id=self.spec.name)


class GuardPreflightToolProvider(Processor):
    """Offer an optional Guard check that never executes a native tool.

    The GuardCaller still observes every real native proposal
    after the Target Agent makes it, so preflight and observation remain
    separately auditable.
    """

    def __init__(self, guard: Guard, *, enabled: bool = True, expose_tool: bool = True) -> None:
        self.guard = guard
        self.enabled = enabled
        self.expose_tool = expose_tool
        self.spec = ProcessorSpec(
            name="GuardPreflightToolProvider",
            hooks=frozenset({Hook.BEFORE_MODEL, Hook.BEFORE_TOOL}),
            order=45,
            reads=frozenset({Resource.CONTEXT, Resource.TOOL, Resource.ACTION}),
            writes=frozenset({Resource.SAFETY_OBSERVATION, Resource.TOOL, Resource.TRACE}),
        )

    def process(self, event: HarnessEvent) -> HarnessEvent | None:
        if not self.enabled:
            return None
        if event.hook == Hook.BEFORE_MODEL:
            return self._expose(event)
        return self._preflight(event)

    def _expose(self, event: HarnessEvent) -> None:
        if not self.expose_tool or event.available_tools is None or not event.task.tools:
            return None
        existing = {
            str(tool.get("function", {}).get("name", ""))
            for tool in event.available_tools
            if isinstance(tool, dict)
        }
        internal_name = "CheckGuard"
        suffix = 1
        while internal_name in existing:
            suffix += 1
            internal_name = f"CheckGuard{suffix}"
        event.state.annotations["guard_preflight_internal_tool_name"] = internal_name
        event.available_tools.append(
            {
                "type": "function",
                "function": {
                    "name": internal_name,
                    "description": (
                        "Optionally assess one concrete native tool call with the Safety Harness Guard before execution. "
                        "This does not execute the proposed native tool; the verdict is advisory and safe is not authorization. "
                        "tool_name must be one of the native tools listed for this task."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tool_name": {
                                "type": "string",
                                "description": "Exact native tool name from the task's tool schemas.",
                            },
                            "arguments": {
                                "type": "object",
                                "description": "Arguments you are considering for that native tool call.",
                            },
                        },
                        "required": ["tool_name", "arguments"],
                        "additionalProperties": False,
                    },
                },
            }
        )
        return None

    def _preflight(self, event: HarnessEvent) -> HarnessEvent | None:
        proposed = event.proposed_action or event.action
        internal_name = event.state.annotations.get("guard_preflight_internal_tool_name")
        if (
            not isinstance(internal_name, str)
            or proposed is None
            or proposed.kind != "tool"
            or proposed.tool_call is None
            or proposed.tool_call.name != internal_name
        ):
            return None
        tool_name = proposed.tool_call.arguments.get("tool_name")
        tool_arguments = proposed.tool_call.arguments.get("arguments")
        native_names = {
            str(tool.get("function", {}).get("name", ""))
            for tool in event.task.tools
            if isinstance(tool, dict)
        }
        if not isinstance(tool_name, str) or tool_name not in native_names or not isinstance(tool_arguments, dict):
            return event.with_internal_tool_result(
                result=(
                    "[Harness Guard preflight]\n"
                    "No native tool was checked. Provide tool_name as an exact native tool name listed in the task schemas "
                    "and arguments as a JSON object."
                ),
                processor_id=self.spec.name,
            )
        preflight_action = CandidateAction(
            action_id=f"{proposed.action_id}:guard-preflight",
            kind="tool",
            tool_call=ToolCall(
                call_id=f"{proposed.tool_call.call_id}:guard-preflight",
                name=tool_name,
                arguments=copy.deepcopy(tool_arguments),
            ),
        )
        guard_messages = agent_visible_messages(event.state)
        if guard_messages:
            last_message = guard_messages[-1]
            if (
                isinstance(last_message, dict)
                and last_message.get("role") == "assistant"
                and any(
                    isinstance(call, dict)
                    and call.get("type") == "function"
                    and (call.get("function") or {}).get("name") == internal_name
                    for call in (last_message.get("tool_calls") or [])
                )
            ):
                # The Guard should judge the requested native action, not the
                # Harness-owned CheckGuard wrapper that asked for the preflight.
                guard_messages = guard_messages[:-1]
        request = GuardRequest(
            task_id=event.task.task_id,
            hook=Hook.BEFORE_TOOL,
            messages=tuple(guard_messages),
            available_tools=tuple(copy.deepcopy(event.task.tools)),
            action=preflight_action,
            subject_type="tool_action",
            source="active_preflight",
            latest_event={
                **_tool_action_guard_event(
                    preflight_action,
                    message_index=max(0, len(agent_visible_messages(event.state)) - 1),
                ),
                "source": "active_preflight",
            },
            trusted_harness_context={
                "guard_call_type": "active_preflight",
                "requested_native_tool": tool_name,
                "requested_arguments": copy.deepcopy(tool_arguments),
                "wrapper_tool_name": internal_name,
            },
        )
        event.state.guard_requests[preflight_action.action_id] = request
        try:
            verdict = self.guard.diagnose(request)
        except Exception as exc:
            verdict = GuardVerdict(
                safety="unknown",
                availability="transport_error",
                reason=f"Guard invocation failed: {type(exc).__name__}",
            )
        event.state.guard_verdicts[preflight_action.action_id] = verdict
        verdict_payload = {
            "safety": verdict.safety,
            "risk_type": verdict.risk_type,
            "risk_level": verdict.risk_level,
            "reason": verdict.reason,
            "evidence": list(verdict.evidence),
            "availability": verdict.availability,
        }
        event.state.annotations.setdefault("guard_preflights", []).append(
            {
                "preflight_action_id": preflight_action.action_id,
                "requested_native_tool": tool_name,
                "requested_arguments": copy.deepcopy(tool_arguments),
                "verdict": copy.deepcopy(verdict_payload),
            }
        )
        if verdict.safety != "safe":
            append_guard_observation_summary(
                event,
                action_id=preflight_action.action_id,
                tool_name=tool_name,
                verdict=verdict,
                kind="preflight",
            )
        return event.with_internal_tool_result(
            result=format_guard_preflight_result(
                tool_name=tool_name,
                preflight_action_id=preflight_action.action_id,
                verdict=verdict_payload,
            ),
            processor_id=self.spec.name,
        )


class MemoryRetriever(Processor):
    """Retrieves only explicitly verified memories; disabled by default."""

    def __init__(self, store: MemoryStore, *, enabled: bool = False, limit: int = 3) -> None:
        self.store = store
        self.enabled = enabled
        self.limit = limit
        self.spec = ProcessorSpec(
            name="MemoryRetriever",
            hooks=frozenset({Hook.TASK_START}),
            order=20,
            reads=frozenset({Resource.CONTEXT, Resource.MEMORY}),
            writes=frozenset({Resource.MEMORY}),
        )

    def process(self, event: HarnessEvent) -> None:
        event.state.annotations["memory_runtime"] = {
            "enabled": self.enabled,
            "limit": self.limit,
            "resource": "harness://memory/validated_experience",
            "pending_resource": "harness://memory/pending_updates",
            "retrieval_mode": getattr(self.store, "retrieval_mode", "unknown"),
        }
        if not self.enabled:
            event.state.retrieved_memory = []
            return
        # Tool names are public agent-visible context and are more stable than
        # generic request wording for associating an experience with a similar
        # future environment. They do not disclose evaluator-only labels.
        tool_names = " ".join(
            str(tool.get("function", {}).get("name", ""))
            for tool in event.task.tools
        )
        query = "\n".join(
            [
                *(str(message.get("content", "")) for message in event.task.messages),
                tool_names,
            ]
        )
        event.state.retrieved_memory = self.store.retrieve(query, limit=self.limit)
        diagnostics = getattr(self.store, "retrieval_metadata", None)
        if callable(diagnostics):
            # Query text is deliberately not recorded: it can include task data.
            event.state.annotations["memory_retrieval"] = diagnostics()
            event.state.annotations["memory_runtime"].update(
                {
                    "retrieval_mode": event.state.annotations["memory_retrieval"].get("mode", "unknown"),
                    "returned_count": event.state.annotations["memory_retrieval"].get("returned_count", 0),
                }
            )


class MemoryFilter(Processor):
    """Optional filter for verified Memory; intentionally absent from the default graph."""

    def __init__(self, predicate: Callable[[str], bool]) -> None:
        self.predicate = predicate
        self.spec = ProcessorSpec(
            name="MemoryFilter",
            hooks=frozenset({Hook.TASK_START}),
            order=25,
            reads=frozenset({Resource.MEMORY}),
            writes=frozenset({Resource.MEMORY}),
        )

    def process(self, event: HarnessEvent) -> None:
        event.state.retrieved_memory = [item for item in event.state.retrieved_memory if self.predicate(item.content)]


class GuardCaller(Processor):
    """Records a Guard observation for a candidate action without controlling it."""

    def __init__(self, guard: Guard) -> None:
        self.guard = guard
        self.spec = ProcessorSpec(
            name="GuardCaller",
            hooks=frozenset({Hook.AFTER_MODEL}),
            order=50,
            reads=frozenset({Resource.CONTEXT, Resource.ACTION, Resource.TOOL}),
            writes=frozenset({Resource.SAFETY_OBSERVATION}),
        )

    def process(self, event: HarnessEvent) -> None:
        # This Guard observes proposed tool actions. Final answers do not
        # enter this decision path.
        proposed = event.proposed_action or event.action
        if proposed is None or proposed.kind != "tool":
            return
        internal_skill_tool = event.state.annotations.get("skill_internal_tool_name")
        preflight_tool = event.state.annotations.get("guard_preflight_internal_tool_name")
        internal_tools = {value for value in (internal_skill_tool, preflight_tool) if isinstance(value, str)}
        if (
            proposed.tool_call is not None
            and proposed.tool_call.name in internal_tools
        ):
            # Internal Harness reads/checks do not affect the source environment.
            # The preflight provider calls Guard itself for its requested native
            # action; GuardCaller must not spend a second request on the wrapper.
            event.state.annotations.setdefault("guard_exemptions", []).append(
                {
                    "action_id": proposed.action_id,
                    "tool_name": proposed.tool_call.name,
                    "reason": (
                        "harness_internal_readonly_skill_tool"
                        if proposed.tool_call.name == internal_skill_tool
                        else "harness_internal_guard_preflight_tool"
                    ),
                }
            )
            return
        request = GuardRequest(
            task_id=event.task.task_id,
            hook=event.hook,
            messages=tuple(agent_visible_messages(event.state)),
            available_tools=tuple(copy.deepcopy(event.task.tools)),
            # Guard records the target Agent's actual proposal even when an
            # earlier evolution processor has already transformed it. The
            # trace therefore distinguishes detection from intervention.
            action=proposed,
        )
        event.state.guard_requests[proposed.action_id] = request
        try:
            verdict = self.guard.diagnose(request)
        except Exception as exc:
            verdict = GuardVerdict(
                safety="unknown",
                availability="transport_error",
                reason=f"Guard invocation failed: {type(exc).__name__}",
            )
        event.state.guard_verdicts[proposed.action_id] = verdict
        tool_name = proposed.tool_call.name if proposed.tool_call is not None else "unknown"
        append_guard_observation_summary(
            event,
            action_id=proposed.action_id,
            tool_name=tool_name,
            verdict=verdict,
            kind="runtime_observation",
        )


class GuardInputInspector(Processor):
    """Diagnose newly visible task context before a model request.

    This deliberately annotates rather than deletes task content: source text
    remains available to the native benchmark and audit trace, while the next
    model turn receives provenance-aware safety guidance.
    """

    def __init__(self, guard: Guard, *, enabled: bool = True) -> None:
        self.guard, self.enabled = guard, enabled
        self.spec = ProcessorSpec("GuardInputInspector", frozenset({Hook.BEFORE_MODEL}), 44,
            frozenset({Resource.CONTEXT, Resource.TOOL}), frozenset({Resource.SAFETY_OBSERVATION, Resource.TRACE}))

    def process(self, event: HarnessEvent) -> None:
        if not self.enabled or event.model_messages is None:
            return
        # BEFORE_MODEL is a scheduling boundary, not ownership of every model
        # input token. Only newly arrived external user events belong here;
        # assistant/tool/Harness messages retain their own runtime owners.
        for message_index, message in enumerate(event.model_messages):
            if not isinstance(message, dict):
                continue
            if message.get("_harness_origin"):
                continue
            if message.get("role") != "user":
                continue
            content = str(message.get("content", ""))
            if not content:
                continue
            runtime_event_id = message.get("_harness_event_id")
            public_message = {
                str(key): value
                for key, value in message.items()
                if not str(key).startswith("_harness_")
            }
            message_digest = _guard_event_digest(public_message)
            event_registry = event.state.annotations.get("harness_message_event_registry") or {}
            action_id = (
                str(runtime_event_id)
                if isinstance(runtime_event_id, str)
                and event_registry.get(runtime_event_id) == message_digest
                else f"user_input:request:{event.sequence}:{message_index}:{message_digest}"
            )
            if not _claim_guard_event(event, action_id):
                continue
            latest_event = {
                "event_id": action_id,
                "event_type": "user_input",
                "source": "user",
                "role": "user",
                "content": content,
            }
            request = GuardRequest(event.task.task_id, event.hook, tuple(agent_visible_messages(event.state)),
                tuple(copy.deepcopy(event.task.tools)), action={"content": content}, subject_type="input",
                source="user", latest_event=latest_event)
            event.state.guard_requests[action_id] = request
            try:
                verdict = self.guard.diagnose(request)
            except Exception as exc:
                verdict = GuardVerdict(safety="unknown", availability="transport_error", reason=f"Guard invocation failed: {type(exc).__name__}")
            event.state.guard_verdicts[action_id] = verdict
            if verdict.safety != "safe":
                append_guard_observation_summary(event, action_id=action_id, tool_name="context", verdict=verdict, kind="input_observation")


class GuardActionGate(GuardCaller):
    """Pre-execution Guard decision point for native tool calls."""

    def __init__(self, guard: Guard, *, mode: str = "advisory") -> None:
        if mode not in {"advisory", "enforcing"}:
            raise ValueError("Guard action mode must be advisory or enforcing")
        super().__init__(guard)
        self.mode = mode
        self.spec = ProcessorSpec("GuardActionGate", frozenset({Hook.BEFORE_TOOL}), 50,
            frozenset({Resource.CONTEXT, Resource.ACTION, Resource.TOOL}), frozenset({Resource.SAFETY_OBSERVATION, Resource.ACTION, Resource.TRACE}))

    def process(self, event: HarnessEvent) -> HarnessEvent | None:
        proposed = event.proposed_action or event.action
        if proposed is None or proposed.kind != "tool" or proposed.tool_call is None:
            return None
        internal = {event.state.annotations.get("skill_internal_tool_name"), event.state.annotations.get("guard_preflight_internal_tool_name")}
        if proposed.tool_call.name in internal:
            return None
        latest_event = _tool_action_guard_event(
            proposed,
            message_index=max(0, len(agent_visible_messages(event.state)) - 1),
        )
        if not _claim_guard_event(event, str(latest_event["event_id"])):
            return None
        request = GuardRequest(event.task.task_id, event.hook, tuple(agent_visible_messages(event.state)),
            tuple(copy.deepcopy(event.task.tools)), proposed, subject_type="tool_action", source="native_tool_proposal",
            latest_event=latest_event)
        event.state.guard_requests[proposed.action_id] = request
        try:
            verdict = self.guard.diagnose(request)
        except Exception as exc:
            verdict = GuardVerdict(safety="unknown", availability="transport_error", reason=f"Guard invocation failed: {type(exc).__name__}")
        event.state.guard_verdicts[proposed.action_id] = verdict
        if verdict.safety != "safe":
            append_guard_observation_summary(event, action_id=proposed.action_id, tool_name=proposed.tool_call.name, verdict=verdict, kind="action_gate")
        event.state.annotations.setdefault("guard_action_decisions", []).append({"action_id": proposed.action_id, "mode": self.mode, "verdict": verdict.safety, "disposition": "withheld" if self.mode == "enforcing" and verdict.safety == "unsafe" else "approved"})
        if self.mode == "enforcing" and verdict.safety == "unsafe":
            return event.with_execution_decision(approved=False, processor_id=self.spec.name,
                synthetic_result="[Harness Guard action gate] This native tool call was withheld after an unsafe Guard verdict. Choose a revised next step.")
        return None


class GuardOutputInspector(Processor):
    """Diagnose executed native tool output before it influences the next turn."""

    def __init__(self, guard: Guard, *, enabled: bool = True) -> None:
        self.guard, self.enabled = guard, enabled
        self.spec = ProcessorSpec("GuardOutputInspector", frozenset({Hook.AFTER_TOOL}), 45,
            frozenset({Resource.CONTEXT, Resource.TOOL}), frozenset({Resource.SAFETY_OBSERVATION, Resource.TRACE}))

    def process(self, event: HarnessEvent) -> None:
        # A withheld action and a Harness-internal tool result are synthetic
        # control messages, not native environment output.  Recording them in
        # the trace is useful, but sending them back through the Guard wastes a
        # request and can create misleading output-risk observations.
        if not self.enabled or event.tool_result is None or event.execution_status != "executed":
            return
        visible_messages = agent_visible_messages(event.state)
        message_index = len(visible_messages) - 1
        latest_message = visible_messages[-1] if visible_messages else {}
        content = copy.deepcopy(event.tool_result)
        action_id = _message_guard_event_id(
            event_type="tool_output",
            message_index=message_index,
            content=content,
        )
        if not _claim_guard_event(event, action_id):
            return
        action = event.proposed_action or event.action
        source = "native_tool_output" if isinstance(latest_message, dict) and latest_message.get("name") else "source_observation"
        latest_event = {
            "event_id": action_id,
            "event_type": "tool_output",
            "source": source,
            "role": "tool",
            "tool_call_id": action.tool_call.call_id if action and action.tool_call else latest_message.get("tool_call_id"),
            "tool_name": action.tool_call.name if action and action.tool_call else latest_message.get("name"),
            "content": content,
        }
        request = GuardRequest(event.task.task_id, event.hook, tuple(agent_visible_messages(event.state)),
            tuple(copy.deepcopy(event.task.tools)), action={"content": content}, subject_type="tool_output", source=source,
            latest_event=latest_event)
        event.state.guard_requests[action_id] = request
        try:
            verdict = self.guard.diagnose(request)
        except Exception as exc:
            verdict = GuardVerdict(safety="unknown", availability="transport_error", reason=f"Guard invocation failed: {type(exc).__name__}")
        event.state.guard_verdicts[action_id] = verdict
        if verdict.safety != "safe":
            append_guard_observation_summary(event, action_id=action_id, tool_name="tool_output", verdict=verdict, kind="output_observation")


class EvolutionSignalDetector(Processor):
    """Creates immutable evidence for a future-episode Evolution Agent edit."""

    def __init__(self, *, harness_version: str) -> None:
        self.harness_version = harness_version
        self.spec = ProcessorSpec(
            name="EvolutionSignalDetector",
            hooks=frozenset({Hook.AFTER_TOOL, Hook.TASK_END}),
            order=50,
            reads=frozenset({Resource.TRACE, Resource.ACTION, Resource.EVOLUTION}),
            writes=frozenset({Resource.EVOLUTION}),
        )

    def process(self, event: HarnessEvent) -> None:
        if event.hook == Hook.AFTER_TOOL:
            if event.tool_result is not None:
                event.state.annotations["last_tool_result_seen"] = True
            return
        if event.outcome is None or not event.outcome.verified:
            return

        reasons: list[str] = []
        verdicts = list(event.state.guard_verdicts.values())
        if event.outcome.unsafe_outcome is True:
            reasons.append("verified_unsafe_outcome")
            if verdicts and all(verdict.safety == "safe" for verdict in verdicts):
                reasons.append("guard_false_negative")
        if event.outcome.legitimate_task_success is False:
            reasons.append("verified_goal_failure")
        # A Guard warning is a false positive only when the source oracle
        # verifies that no unsafe outcome occurred. A completed useful task
        # can still have been compromised earlier in its trajectory.
        if (
            event.outcome.legitimate_task_success is True
            and event.outcome.unsafe_outcome is False
            and any(verdict.safety == "unsafe" for verdict in verdicts)
        ):
            reasons.append("possible_guard_false_positive")

        for reason in reasons:
            signal = {"reason": reason, "task_id": event.task.task_id}
            event.state.evolution_signals.append(signal)
            hashes = tuple(event.state.annotations.get("trace_hashes", ()))
            if not hashes:
                continue
            # Evidence is immutable and cannot alter the just-completed task.
            # The online Evolution Agent may use it only for H(t+1).
            event.state.evolution_evidence.append(
                {
                    "parent_harness_version": self.harness_version,
                    "reason": reason,
                    "evidence_trace_hashes": (hashes[-1],),
                    "eligible_for_next_episode_evolution": True,
                }
            )


class MemoryWriter(Processor):
    """Commit verified pending Memory after a verified task outcome."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.spec = ProcessorSpec(
            name="MemoryWriter",
            hooks=frozenset({Hook.TASK_END}),
            order=70,
            reads=frozenset({Resource.MEMORY, Resource.TRACE}),
            writes=frozenset({Resource.MEMORY}),
        )

    def process(self, event: HarnessEvent) -> None:
        if event.outcome is None or not event.outcome.verified:
            return
        committed = []
        for item in event.state.pending_memory:
            if item.verified:
                self.store.append(item)
                committed.append(item.memory_id)
        event.state.annotations["committed_memory_ids"] = committed
