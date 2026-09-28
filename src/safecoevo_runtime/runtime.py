"""Hook dispatcher and session API for the evolvable Safety Harness."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
from collections import defaultdict, deque
from dataclasses import dataclass

from .contracts import CandidateAction, GuardVerdict, Hook, TaskOutcome
from .events import HarnessEvent, HarnessState, PublicTask, agent_visible_messages, agent_visible_observations, strip_harness_message_metadata
from .memory import InMemoryMemoryStore, MemoryStore
from .permission import InMemoryPermissionExperienceStore, PermissionExperienceStore
from .processor import Processor
from .processors import (
    CallableGuard,
    EvolutionSignalDetector,
    Guard,
    GuardActionGate,
    GuardCaller,
    GuardInputInspector,
    GuardOutputInspector,
    GuardPreflightToolProvider,
    MemoryRetriever,
    MemoryWriter,
    PermissionExperienceInjector,
    PromptFormatter,
    ProgressiveSkillLoader,
    SkillCatalogProcessor,
    SkillRuntimeProcessor,
    SkillToolProvider,
    TraceRecorder,
)
from .skills import SkillRegistry
from .target_prompts import (
    format_execution_record_appendix,
    format_guard_observation_appendix,
    format_processor_observation_block,
)
from .trace import InMemoryTraceStore, TraceStore


@dataclass(frozen=True)
class HarnessConfig:
    version: str


@dataclass(frozen=True)
class ActionExecution:
    """The auditable result of routing one target-Agent proposal.

    ``proposed_action`` is always the target model's raw output. The effective
    action may differ after one or more Processor transformations. A withheld
    action has no environment execution; a synthetic result, when present,
    is returned through the normal tool-result channel by the source adapter.
    ``internal`` identifies a Harness-owned tool such as ``LoadSkill``: its
    result is real local data but must never enter a source executor.
    """

    proposed_action: CandidateAction
    effective_action: CandidateAction | None
    status: str
    synthetic_tool_result: str | None = None
    intercepted_by: tuple[str, ...] = ()


class HarnessBuilder:
    """Register processors and build a Safety Harness runtime."""

    def __init__(self, *, config: HarnessConfig) -> None:
        self.config = config
        self._processors: list[Processor] = []

    def add(self, processor: Processor) -> "HarnessBuilder":
        self._processors.append(processor)
        return self

    def build(self) -> "SafetyHarness":
        names = [processor.spec.name for processor in self._processors]
        if len(names) != len(set(names)):
            raise ValueError(f"Processor names must be unique: {names}")
        routed: dict[Hook, list[Processor]] = defaultdict(list)
        for processor in self._processors:
            if not processor.spec.hooks:
                raise ValueError(f"{processor.spec.name} has no registered Hook")
            for hook in processor.spec.hooks:
                routed[hook].append(processor)
        for hook, processors in routed.items():
            routed[hook] = _order_processors(processors, hook)
        return SafetyHarness(config=self.config, processors=dict(routed))


class SafetyHarness:
    """Routes lifecycle events through ordered, side-effect-scoped Processors."""

    def __init__(self, *, config: HarnessConfig, processors: dict[Hook, list[Processor]]) -> None:
        self.config = config
        self._processors = processors
        self._config_hash = self._hash_config()

    def _hash_config(self) -> str:
        payload = {
            "version": self.config.version,
            "processors": {
                hook.value: [
                    {
                        "name": processor.spec.name,
                        "order": processor.spec.order,
                        "reads": sorted(resource.value for resource in processor.spec.reads),
                        "writes": sorted(resource.value for resource in processor.spec.writes),
                        "after": sorted(processor.spec.after),
                    }
                    for processor in processors
                ]
                for hook, processors in sorted(self._processors.items(), key=lambda item: item[0].value)
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def config_hash(self) -> str:
        return self._config_hash

    def processors_for(self, hook: Hook) -> tuple[Processor, ...]:
        return tuple(self._processors.get(hook, ()))

    def start(self, task: PublicTask) -> "HarnessSession":
        # ``PublicTask.system`` is deliberately separate in the unified data
        # schema. Seed the complete model-visible conversation before any
        # Processor runs; dropping it would change the source task.
        # ``_harness_*`` is a runtime-owned namespace. Strip any source-supplied
        # lookalikes before the Harness assigns provenance to its own messages.
        initial_messages = strip_harness_message_metadata(copy.deepcopy(list(task.messages)))
        initial_system = strip_harness_message_metadata([copy.deepcopy(task.system)])[0]
        state = HarnessState(
            # ``messages`` intentionally excludes the separately stored system
            # prompt: PromptFormatter owns assembling one final system message
            # by appending the Safety Harness addendum to ``PublicTask.system``.
            messages=initial_messages,
            formatted_messages=[initial_system, *copy.deepcopy(initial_messages)],
        )
        state.annotations.update(
            {
                "harness_version": self.config.version,
                "harness_config_hash": self.config_hash,
                "active_processors": sorted(
                    {
                        processor.spec.name
                        for processors in self._processors.values()
                        for processor in processors
                    }
                ),
                "guard_runtime": {
                    "enabled": any(
                        processor.spec.name in {
                            "GuardCaller",
                            "GuardPreflightToolProvider",
                            "GuardInputInspector",
                            "GuardActionGate",
                            "GuardOutputInspector",
                            # Graph-instantiated processors expose their
                            # stable node ids as spec names.
                            "guard_caller",
                            "guard_preflight_tool_provider",
                            "guard_input_inspector",
                            "guard_action_gate",
                            "guard_output_inspector",
                        }
                        for processors in self._processors.values()
                        for processor in processors
                    ),
                    "resource": "harness://guard/runtime_observations",
                    "preflight_resource": "harness://guard/check_guard",
                },
            }
        )
        session = HarnessSession(harness=self, task=task, state=state)
        session._dispatch(HarnessEvent(hook=Hook.TASK_START, task=task, state=state))
        session._materialize_processor_observations()
        return session


class HarnessSession:
    """A source executor drives this session around its target-model/tool loop."""

    def __init__(self, *, harness: SafetyHarness, task: PublicTask, state: HarnessState) -> None:
        self.harness = harness
        self.task = task
        self.state = state
        self._sequence = 0
        self._last_action: CandidateAction | None = None
        self._last_execution: ActionExecution | None = None
        self._closed = False

    def _ensure_persistent_message_event_ids(self) -> None:
        """Assign runtime-owned stable IDs to persistent user-role messages."""

        registry = self.state.annotations.setdefault("harness_message_event_registry", {})
        positions = self.state.annotations.setdefault("harness_message_event_positions", {})
        counter = int(self.state.annotations.get("harness_message_event_counter", 0))
        for message_index, message in enumerate(self.state.formatted_messages):
            if message.get("role") != "user":
                continue
            public_message = strip_harness_message_metadata([message])[0]
            encoded = json.dumps(
                public_message,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
            digest = hashlib.sha256(encoded).hexdigest()[:16]
            position_key = str(message_index)
            assigned = positions.get(position_key)
            if isinstance(assigned, str) and registry.get(assigned) == digest:
                message["_harness_event_id"] = assigned
                continue
            # A source-supplied lookalike is not trusted unless this runtime
            # previously issued it in the task-local state.
            message.pop("_harness_event_id", None)
            event_id = f"user_input:{counter}:{digest}"
            while event_id in registry:
                counter += 1
                event_id = f"user_input:{counter}:{digest}"
            message["_harness_event_id"] = event_id
            registry[event_id] = digest
            positions[position_key] = event_id
            counter += 1
        self.state.annotations["harness_message_event_counter"] = counter

    def _dispatch(self, event: HarnessEvent) -> tuple[HarnessEvent, ...]:
        """Route an event through synchronous and event-stream processors.

        Existing builtins mutate an event and return ``None``. Generated
        classes return async event streams. Empty generated output is an
        interception: we preserve the event for downstream audit processors
        while marking its execution status as ``withheld``.
        """

        if self._closed and event.hook != Hook.TASK_END:
            raise RuntimeError("Cannot dispatch to a completed Harness session")
        self._sequence += 1
        event.sequence = self._sequence
        events: tuple[HarnessEvent, ...] = (event,)
        for processor in self.harness.processors_for(event.hook):
            outputs: list[HarnessEvent] = []
            for current in events:
                processed = _run_processor(processor, current)
                if not processed:
                    current.execution_status = "withheld"
                    current.intercepted_by = (*current.intercepted_by, processor.spec.name)
                    current.state.annotations.setdefault("processor_interceptions", []).append(
                        {
                            "processor_id": processor.spec.name,
                            "hook": current.hook.value,
                            "reason": "processor_yielded_no_event",
                        }
                    )
                    # Continue only so TraceRecorder and later audit hooks can
                    # persist the interception. Source adapters inspect the
                    # explicit execution status before touching the tool.
                    processed = (current,)
                outputs.extend(processed)
            events = tuple(outputs)
        return events

    def agent_input(self) -> dict:
        """Return the complete public context plus Harness observations."""

        return {
            "messages": agent_visible_messages(self.state),
            "tools": copy.deepcopy(list(self.task.tools)),
            "runtime": copy.deepcopy(self.task.runtime),
            # Adapters may serialize these as a transparent model-visible
            # context field, but the runtime never turns them into commands.
            "harness_observations": agent_visible_observations(self.state),
            "processor_observations": copy.deepcopy(self.state.processor_observations),
        }

    def prepare_model_request(self) -> dict:
        """Construct one model request after request-local processors run.

        The source task and native tool table remain immutable. A processor
        can add an advisory Skill reference to the task-local Harness history
        exactly once, while the Harness-internal LoadSkill schema exists only
        in this request and can never reach a native executor.
        """

        self._ensure_persistent_message_event_ids()
        event = HarnessEvent(
            hook=Hook.BEFORE_MODEL,
            task=self.task,
            state=self.state,
            model_messages=agent_visible_messages(self.state, include_harness_metadata=True),
            available_tools=copy.deepcopy(list(self.task.tools)),
        )
        events = self._dispatch(event)
        effective = _primary_event(events, event)
        if effective.model_messages is None or effective.available_tools is None:
            raise RuntimeError("before_model processor returned an invalid model request")
        self.state.annotations.setdefault("model_requests", []).append(
            {
                "sequence": effective.sequence,
                "message_count": len(effective.model_messages),
                "tool_names": [
                    str(tool.get("function", {}).get("name", ""))
                    for tool in effective.available_tools
                    if isinstance(tool, dict)
                ],
            }
        )
        return {
            "messages": strip_harness_message_metadata(effective.model_messages),
            "tools": copy.deepcopy(effective.available_tools),
            "runtime": copy.deepcopy(self.task.runtime),
            "harness_observations": agent_visible_observations(self.state),
            "processor_observations": copy.deepcopy(self.state.processor_observations),
        }

    def submit_action(self, action: CandidateAction) -> ActionExecution:
        """Route a raw target proposal and return its executable disposition."""

        if self._closed:
            raise RuntimeError("Cannot submit actions after task_end")
        self._append_agent_action(action)
        self._last_action = action
        event = HarnessEvent(
            hook=Hook.AFTER_MODEL,
            task=self.task,
            state=self.state,
            action=action,
            proposed_action=action,
        )
        after_model = self._dispatch(event)
        effective = _primary_event(after_model, event)
        if action.kind == "tool" and effective.execution_status != "withheld" and effective.action is not None:
            before_tool = self._dispatch(
                HarnessEvent(
                    hook=Hook.BEFORE_TOOL,
                    task=self.task,
                    state=self.state,
                    action=effective.action,
                    proposed_action=action,
                    execution_status=effective.execution_status,
                    synthetic_tool_result=effective.synthetic_tool_result,
                    intercepted_by=effective.intercepted_by,
                )
            )
            effective = _primary_event(before_tool, effective)

        if effective.action is None:
            effective.execution_status = "withheld"
        elif effective.execution_status == "pending":
            effective.execution_status = "approved"
        execution = ActionExecution(
            proposed_action=action,
            effective_action=effective.action if effective.execution_status not in {"withheld"} else None,
            status=effective.execution_status,
            synthetic_tool_result=effective.synthetic_tool_result,
            intercepted_by=effective.intercepted_by,
        )
        self._last_execution = execution
        self.state.annotations.setdefault("action_executions", []).append(
            {
                "proposed_action_id": action.action_id,
                "effective_action_id": execution.effective_action.action_id if execution.effective_action else None,
                "status": execution.status,
                "synthetic_tool_result": execution.synthetic_tool_result,
                "intercepted_by": list(execution.intercepted_by),
            }
        )
        return execution

    def record_tool_result(self, result: object) -> None:
        """Append an executed tool result and expose it to the after-tool Hook."""

        if self._last_action is None or self._last_action.kind != "tool" or self._last_action.tool_call is None:
            raise RuntimeError("A tool result requires a preceding tool candidate action")
        content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        verdict = self.state.guard_verdicts.get(self._last_action.action_id)
        if verdict is not None:
            observation = {
                self._last_action.action_id: {
                    "safety": verdict.safety,
                    "risk_type": verdict.risk_type,
                    "risk_level": verdict.risk_level,
                    "reason": verdict.reason,
                    "evidence": list(verdict.evidence),
                    "availability": verdict.availability,
                }
            }
            # Keep function-call roles valid: short NL nudge + JSON details on
            # the native tool result (not a synthetic user turn).
            content += format_guard_observation_appendix(
                tool_name=self._last_action.tool_call.name,
                action_id=self._last_action.action_id,
                verdict=observation[self._last_action.action_id],
            )
        if self._last_execution is not None and self._last_execution.status != "pending":
            execution_payload = {
                "proposed_action_id": self._last_execution.proposed_action.action_id,
                "effective_action_id": self._last_execution.effective_action.action_id if self._last_execution.effective_action else None,
                "status": self._last_execution.status,
                "intercepted_by": list(self._last_execution.intercepted_by),
            }
            content += format_execution_record_appendix(execution_payload)
        tool_message = {
            "role": "tool",
            "tool_call_id": self._last_action.tool_call.call_id,
            "name": self._last_action.tool_call.name,
            "content": content,
        }
        self.state.messages.append(copy.deepcopy(tool_message))
        self.state.formatted_messages.append(copy.deepcopy(tool_message))
        self._dispatch(
            HarnessEvent(
                hook=Hook.AFTER_TOOL,
                task=self.task,
                state=self.state,
                action=(self._last_execution.effective_action if self._last_execution and self._last_execution.effective_action else self._last_action),
                proposed_action=self._last_action,
                execution_status=(
                    "executed"
                    if self._last_execution and self._last_execution.status == "approved"
                    else self._last_execution.status if self._last_execution else "executed"
                ),
                synthetic_tool_result=self._last_execution.synthetic_tool_result if self._last_execution else None,
                intercepted_by=self._last_execution.intercepted_by if self._last_execution else (),
                tool_result=copy.deepcopy(result),
            )
        )
        # AFTER_TOOL Processors may add observations. Attach only the newly
        # created data to this native tool result so function-call history
        # remains assistant -> tool -> assistant and no action is rewritten.
        self._materialize_processor_observations(tool_result=True)

    def record_source_observation(self, content: object) -> None:
        """Append a source-scheduled post-tool observation without controlling it.

        Some native environments deliver a second tool-like observation after
        an otherwise normal tool result. This preserves that protocol while
        allowing ``after_tool`` Processors to see the delivered data before
        the target Agent's next model request.
        """

        if self._last_action is None or self._last_action.kind != "tool" or self._last_action.tool_call is None:
            raise RuntimeError("A source observation requires a preceding tool candidate action")
        value = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
        message = {
            "role": "tool",
            "tool_call_id": self._last_action.tool_call.call_id,
            "content": value,
        }
        self.state.messages.append(copy.deepcopy(message))
        self.state.formatted_messages.append(copy.deepcopy(message))
        self._dispatch(
            HarnessEvent(
                hook=Hook.AFTER_TOOL,
                task=self.task,
                state=self.state,
                action=(self._last_execution.effective_action if self._last_execution and self._last_execution.effective_action else self._last_action),
                proposed_action=self._last_action,
                execution_status=(
                    "executed"
                    if self._last_execution and self._last_execution.status == "approved"
                    else self._last_execution.status if self._last_execution else "executed"
                ),
                synthetic_tool_result=self._last_execution.synthetic_tool_result if self._last_execution else None,
                intercepted_by=self._last_execution.intercepted_by if self._last_execution else (),
                tool_result=copy.deepcopy(content),
            )
        )
        self._materialize_processor_observations(tool_result=True)

    def add_verified_memory(self, item: object) -> None:
        """Queue a MemoryItem for task-end commit; validation happens in MemoryWriter."""

        from .contracts import MemoryItem

        if not isinstance(item, MemoryItem):
            raise TypeError("Only MemoryItem values may be queued for long-term memory")
        self.state.pending_memory.append(item)

    def end(self, outcome: TaskOutcome) -> HarnessEvent:
        """Close the task and permit signal generation plus verified Memory writes."""

        if self._closed:
            raise RuntimeError("task_end may only be emitted once")
        event = self._dispatch(HarnessEvent(hook=Hook.TASK_END, task=self.task, state=self.state, outcome=outcome))
        self._closed = True
        return _primary_event(event, HarnessEvent(hook=Hook.TASK_END, task=self.task, state=self.state, outcome=outcome))

    def _append_agent_action(self, action: CandidateAction) -> None:
        if action.kind == "final":
            message = {"role": "assistant", "content": action.text}
        else:
            assert action.tool_call is not None
            message = {
                "role": "assistant",
                "content": action.text,
                "tool_calls": [
                    {
                        "id": action.tool_call.call_id,
                        "type": "function",
                        "function": {
                            "name": action.tool_call.name,
                            "arguments": json.dumps(action.tool_call.arguments, ensure_ascii=False, sort_keys=True),
                        },
                    }
                ],
            }
        self.state.messages.append(copy.deepcopy(message))
        self.state.formatted_messages.append(copy.deepcopy(message))

    def _materialize_processor_observations(self, *, start: int | None = None, tool_result: bool = False) -> None:
        """Expose generated Processor output as labeled advisory notes."""

        delivered = self.state.annotations.get("processor_observations_delivered", 0)
        if not isinstance(delivered, int) or delivered < 0:
            delivered = 0
        first = delivered if start is None else start
        observations = self.state.processor_observations[first:]
        if not observations:
            return
        block = format_processor_observation_block(list(observations))
        if tool_result:
            for messages in (self.state.messages, self.state.formatted_messages):
                if messages and messages[-1].get("role") == "tool":
                    messages[-1]["content"] = str(messages[-1].get("content", "")) + "\n\n" + block
            self.state.annotations["processor_observations_delivered"] = len(self.state.processor_observations)
            return
        message = {
            "role": "user",
            "content": block,
            "_harness_origin": "processor_observation",
        }
        self.state.messages.append(copy.deepcopy(message))
        self.state.formatted_messages.append(copy.deepcopy(message))
        self.state.annotations["processor_observations_delivered"] = len(self.state.processor_observations)


def _primary_event(events: tuple[HarnessEvent, ...], fallback: HarnessEvent) -> HarnessEvent:
    """Select the primary event after a Hook chain.

    Multi-event processors are valid for audit and future orchestration
    extensions. A source adapter can execute only one target action per turn,
    so the last emitted event is the effective one for that boundary.
    """

    return events[-1] if events else fallback


def _run_processor(processor: Processor, event: HarnessEvent) -> tuple[HarnessEvent, ...]:
    """Normalize synchronous and asynchronous processor outputs."""

    value = processor.process(event)
    if inspect.isasyncgen(value):
        async def collect() -> tuple[HarnessEvent, ...]:
            items: list[HarnessEvent] = []
            async for item in value:
                items.append(item)
            return tuple(items)

        return asyncio.run(collect())
    if inspect.isawaitable(value):
        resolved = asyncio.run(value)
        if resolved is None:
            return (event,)
        if isinstance(resolved, HarnessEvent):
            return (resolved,)
        raise TypeError(
            f"{type(processor).__name__}.process returned unsupported awaitable result "
            f"{type(resolved).__name__}"
        )
    if value is None:
        return (event,)
    if isinstance(value, HarnessEvent):
        return (value,)
    if isinstance(value, (tuple, list)) and all(isinstance(item, HarnessEvent) for item in value):
        return tuple(value)
    raise TypeError(
        f"{type(processor).__name__}.process returned unsupported result {type(value).__name__}"
    )


def _order_processors(processors: list[Processor], hook: Hook) -> list[Processor]:
    """Order a Hook's Processor chain by ``order`` then ``after`` edges."""

    by_name = {processor.spec.name: processor for processor in processors}
    if len(by_name) != len(processors):  # Defensive: builder checks globally.
        raise ValueError(f"Duplicate Processor names on hook {hook.value}")
    successors: dict[str, set[str]] = {name: set() for name in by_name}
    indegree: dict[str, int] = {name: 0 for name in by_name}
    for name, processor in by_name.items():
        for predecessor in processor.spec.after:
            target = by_name.get(predecessor)
            if target is None:
                # A multi-Hook Processor may depend on another component only
                # on a subset of its Hooks. The graph validator ensures the
                # named component exists somewhere in the graph; this Hook
                # simply has no local ordering edge when it is absent here.
                continue
            if target.spec.order > processor.spec.order:
                raise ValueError(
                    f"Processor {name} order={processor.spec.order} cannot run after {predecessor} "
                    f"order={target.spec.order} on hook {hook.value}"
                )
            if name not in successors[predecessor]:
                successors[predecessor].add(name)
                indegree[name] += 1

    ready = deque(
        sorted(
            (name for name, degree in indegree.items() if degree == 0),
            key=lambda name: (by_name[name].spec.order, name),
        )
    )
    ordered: list[Processor] = []
    while ready:
        name = ready.popleft()
        ordered.append(by_name[name])
        newly_ready: list[str] = []
        for successor in sorted(successors[name]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                newly_ready.append(successor)
        if newly_ready:
            ready.extend(
                sorted(newly_ready, key=lambda item: (by_name[item].spec.order, item))
            )
            ready = deque(sorted(ready, key=lambda item: (by_name[item].spec.order, item)))
    if len(ordered) != len(processors):
        cyclic = sorted(name for name, degree in indegree.items() if degree > 0)
        raise ValueError(f"Cycle in Processor after dependencies on hook {hook.value}: {cyclic}")
    return ordered


def build_minimal_harness(
    *,
    guard: Guard | None = None,
    trace_store: TraceStore | None = None,
    memory_store: MemoryStore | None = None,
    memory_enabled: bool = False,
    memory_limit: int = 3,
    permission_store: PermissionExperienceStore | None = None,
    permission_enabled: bool = True,
    permission_limit: int = 3,
    guard_enabled: bool = True,
    guard_preflight_tool: bool = True,
    guard_action_mode: str = "advisory",
    skill_registry: SkillRegistry | None = None,
    prompt_addendum_template: str | None = None,
    guard_policy: dict[str, object] | None = None,
    skill_enabled: bool = False,
    skill_auto_inject: bool = True,
    skill_load_tool: bool = True,
    skill_top_k: int = 3,
    skill_max_active: int = 8,
    version: str = "safecoevo-runtime",
) -> SafetyHarness:
    """Build the documented minimum graph without auto-deploying evolution edits."""

    if guard is None:
        guard = CallableGuard(lambda _request: GuardVerdict(safety="safe"))
    if trace_store is None:
        trace_store = InMemoryTraceStore()
    if memory_store is None:
        memory_store = InMemoryMemoryStore()
    if permission_store is None:
        permission_store = InMemoryPermissionExperienceStore()
    if skill_registry is None:
        skill_registry = SkillRegistry()
    config = HarnessConfig(version=version)
    builder = (
        HarnessBuilder(config=config)
        .add(MemoryRetriever(memory_store, enabled=memory_enabled, limit=memory_limit))
        .add(SkillCatalogProcessor(skill_registry, enabled=skill_enabled, max_active=skill_max_active))
        .add(SkillRuntimeProcessor(skill_registry, enabled=skill_enabled))
        .add(PermissionExperienceInjector(permission_store, enabled=permission_enabled, limit=permission_limit))
    )
    builder = (
        builder
        .add(PromptFormatter(addendum_template=prompt_addendum_template, guard_policy=guard_policy))
        .add(ProgressiveSkillLoader(skill_registry, enabled=skill_enabled and skill_auto_inject, top_k=skill_top_k))
        .add(SkillToolProvider(skill_registry, enabled=skill_enabled, expose_tool=skill_load_tool))
        .add(EvolutionSignalDetector(harness_version=version))
        .add(TraceRecorder(trace_store))
    )
    builder.add(MemoryWriter(memory_store))
    if guard_enabled:
        builder.add(GuardInputInspector(guard))
        if guard_preflight_tool:
            builder.add(GuardPreflightToolProvider(guard))
        builder.add(GuardActionGate(guard, mode=guard_action_mode))
        builder.add(GuardOutputInspector(guard))
    return builder.build()
