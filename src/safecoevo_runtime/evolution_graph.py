"""Versioned Processor graphs and model transport for fixed-artifact evolution.

Persisted graphs may contain generated processor classes, whose sources are
validated before a session starts.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import time
import types
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol

from .contracts import Hook, Resource
from .events import HarnessEvent
from .memory import MemoryStore
from .permission import InMemoryPermissionExperienceStore, PermissionExperienceStore
from .processor import MultiHookProcessor, Processor, ProcessorSpec
from .processors import (
    EvolutionSignalDetector,
    GuardActionGate,
    Guard,
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
from .trace import TraceStore
from .evolution_workspace import EvolutionWorkspace, EvolutionWorkspaceTools
from .artifact_patcher import artifact_patch_template
from .prompt_templates import render_prompt_template


GRAPH_SCHEMA_VERSION = 1
MAX_SOURCE_BYTES = 24_000
# A conservative byte ceiling keeps the direct evidence payload well below
# providers whose context limits are expressed in tokens. Larger trajectories
# remain exact in the workspace and are read record-by-record on demand.
MAX_DIRECT_EVOLUTION_EVIDENCE_BYTES = 8_000_000


def _thinking_disabled_extra_body(model: str, base_url: str = "") -> dict[str, Any] | None:
    """Return provider-specific non-thinking knob only for endpoints known to accept it."""

    marker = f"{model} {base_url}".lower()
    if "gpt-" in marker or "kimi" in marker:
        return None
    if "qwen" in marker:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    if "dsv4" in marker or "ds-v4" in marker or "deepseek" in marker:
        return {"chat_template_kwargs": {"thinking": False}}
    return None

EvolutionPolicyName = Literal["fixed_artifact_patch"]


@dataclass(frozen=True)
class EvolutionPolicy:
    """Describes the fixed artifact update boundary for an evolution step."""

    name: EvolutionPolicyName

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.name, "allowed_categories": ["prompt", "memory", "skill", "permission", "guard"]}


def evolution_policy(name: str) -> EvolutionPolicy:
    """Return the policy recorded for fixed artifact updates."""

    if name != "fixed_artifact_patch":
        raise ValueError(f"Unknown Evolution policy: {name}")
    return EvolutionPolicy(name="fixed_artifact_patch")


@dataclass(frozen=True)
class ProcessorNode:
    """One named Processor placement in a frozen graph version."""

    processor_id: str
    processor_name: str
    implementation: Literal["builtin", "generated"]
    hooks: tuple[Hook, ...]
    order: int
    parameters: dict[str, Any] = field(default_factory=dict)
    source: str | None = None
    # Generated sources export this exact class name.
    exported_class: str | None = None
    # Ordering dependencies between processors registered on the same hook.
    after: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "processor_id": self.processor_id,
            "processor_name": self.processor_name,
            "implementation": self.implementation,
            "hooks": [hook.value for hook in self.hooks],
            "order": self.order,
            "parameters": copy.deepcopy(self.parameters),
            "source": self.source,
            "exported_class": self.exported_class,
            "after": list(self.after),
        }


@dataclass(frozen=True)
class ProcessorGraph:
    """The Processor topology used for one complete target-Agent episode."""

    version: str
    nodes: tuple[ProcessorNode, ...]
    schema_version: int = GRAPH_SCHEMA_VERSION

    @property
    def graph_hash(self) -> str:
        return _hash({"version": self.version, "nodes": [node.to_dict() for node in self.nodes]})

    def to_dict(self, *, include_source: bool = True) -> dict[str, Any]:
        nodes = []
        for node in self.nodes:
            item = node.to_dict()
            if not include_source:
                source = item.pop("source")
                item["source_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest() if source else None
            nodes.append(item)
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "nodes": nodes,
            "graph_sha256": self.graph_hash,
        }

    def node(self, processor_id: str) -> ProcessorNode:
        for node in self.nodes:
            if node.processor_id == processor_id:
                return node
        raise KeyError(processor_id)


def processor_graph_from_dict(value: dict[str, Any]) -> ProcessorGraph:
    """Restore a persisted graph and reject malformed or tampered versions."""

    if not isinstance(value, dict):
        raise ValueError("Persisted Processor graph must be an object")
    version = value.get("version")
    raw_nodes = value.get("nodes")
    schema_version = value.get("schema_version", GRAPH_SCHEMA_VERSION)
    if not isinstance(version, str) or not version:
        raise ValueError("Persisted Processor graph needs a version")
    if not isinstance(raw_nodes, list):
        raise ValueError("Persisted Processor graph needs a nodes list")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise ValueError("Persisted Processor graph has invalid schema_version")
    nodes: list[ProcessorNode] = []
    for raw in raw_nodes:
        if not isinstance(raw, dict):
            raise ValueError("Persisted Processor node must be an object")
        processor_id = raw.get("processor_id")
        processor_name = raw.get("processor_name")
        implementation = raw.get("implementation")
        hooks = raw.get("hooks")
        order = raw.get("order")
        parameters = raw.get("parameters", {})
        source = raw.get("source")
        exported_class = raw.get("exported_class")
        after = raw.get("after", [])
        if not isinstance(processor_id, str) or not isinstance(processor_name, str):
            raise ValueError("Persisted Processor node needs names")
        if implementation not in {"builtin", "generated"}:
            raise ValueError("Persisted Processor node has invalid implementation")
        if not isinstance(hooks, list) or not hooks:
            raise ValueError("Persisted Processor node needs hooks")
        if not isinstance(order, int) or isinstance(order, bool):
            raise ValueError("Persisted Processor node has invalid order")
        if (
            not isinstance(parameters, dict)
            or source is not None and not isinstance(source, str)
            or exported_class is not None and not isinstance(exported_class, str)
            or not isinstance(after, list) or not all(isinstance(item, str) for item in after)
        ):
            raise ValueError("Persisted Processor node has invalid parameters or source")
        nodes.append(
            ProcessorNode(
                processor_id=processor_id,
                processor_name=processor_name,
                implementation=implementation,
                hooks=tuple(Hook(str(hook)) for hook in hooks),
                order=order,
                parameters=copy.deepcopy(parameters),
                source=source,
                exported_class=exported_class,
                after=tuple(after),
            )
        )
    graph = ProcessorGraph(version=version, nodes=tuple(nodes), schema_version=schema_version)
    validate_graph(graph)
    expected_hash = value.get("graph_sha256")
    if expected_hash is not None and expected_hash != graph.graph_hash:
        raise ValueError("Persisted Processor graph hash does not match its content")
    return graph


@dataclass(frozen=True)
class EvolutionDeployment:
    """One auditable post-episode transaction decision."""

    parent_version: str
    next_version: str
    task_id: str
    candidate_id: str | None
    status: Literal["no_change", "accepted", "rejected", "planner_error", "deferred_batch"]
    reason: str
    applied_edits: tuple[dict[str, Any], ...] = ()
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_version": self.parent_version,
            "next_version": self.next_version,
            "task_id": self.task_id,
            "candidate_id": self.candidate_id,
            "status": self.status,
            "reason": self.reason,
            "applied_edits": copy.deepcopy(list(self.applied_edits)),
            "validation": copy.deepcopy(self.validation),
        }


class EvolutionAgent(Protocol):
    """Proposes a fixed-artifact update without direct runtime writes."""

    def propose(
        self,
        *,
        graph: ProcessorGraph,
        evidence: dict[str, Any],
        workspace: EvolutionWorkspace | None = None,
    ) -> dict[str, Any]: ...


class StaticEvolutionAgent:
    """Deterministic adapter for evolution-disabled and local runs."""

    def __init__(self, proposals: Iterable[dict[str, Any]]) -> None:
        self._proposals = iter(proposals)

    def propose(
        self,
        *,
        graph: ProcessorGraph,
        evidence: dict[str, Any],
        workspace: EvolutionWorkspace | None = None,
    ) -> dict[str, Any]:
        del graph, evidence, workspace
        try:
            return next(self._proposals)
        except StopIteration:
            return {"reason": "No artifact change is justified.", "changes": [], "file_updates": []}


class OpenAICompatibleEvolutionAgent:
    """OpenAI-compatible model adapter for the autonomous Evolution Agent."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key_env: str,
        # Evolution sessions may read several source files and emit a complete
        # artifact patch, so give their final proposal room.
        max_tokens: int = 16384,
        timeout_seconds: float = 240.0,
        request_retries: int = 1,
        fallback_model: str = "",
        fallback_base_url: str = "",
        fallback_api_key: str = "",
        fallback_request_retries: int = 2,
        retry_sleep_seconds: float = 5.0,
        max_tool_rounds: int = 100,
        policy: EvolutionPolicy | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        if request_retries < 0:
            raise ValueError("Evolution Agent request retries must be non-negative")
        self.request_retries = request_retries
        if fallback_request_retries < 0:
            raise ValueError("Evolution Agent fallback request retries must be non-negative")
        if retry_sleep_seconds < 0:
            raise ValueError("Evolution Agent retry sleep must be non-negative")
        self.fallback_model = fallback_model
        self.fallback_base_url = fallback_base_url
        self.fallback_api_key = fallback_api_key
        self.fallback_request_retries = fallback_request_retries
        self.retry_sleep_seconds = retry_sleep_seconds
        if max_tool_rounds < 1:
            raise ValueError("Evolution Agent max_tool_rounds must be positive")
        self.max_tool_rounds = max_tool_rounds
        self.policy = policy or evolution_policy("fixed_artifact_patch")
        self.last_parse_repaired = False
        self.last_session: list[dict[str, Any]] = []
        # Kept separately from the chat transcript so an evaluator can answer
        # exactly which workspace artifacts the model received from each tool.
        self.last_tool_audit: list[dict[str, Any]] = []
        self.last_tool_rounds = 0
        self.last_request_audit: list[dict[str, Any]] = []

    def _chat_completion_with_retry_and_fallback(self, request: dict[str, Any]) -> Any:
        """Call the Evolution model with primary retries, then fixed fallback.

        Tool effects occur only after a response is returned and processed, so
        retrying a failed model request does not replay a completed task or a
        completed workspace mutation.
        """

        from openai import OpenAI

        primary_key = os.environ.get(self.api_key_env)
        if not primary_key:
            raise RuntimeError(f"Missing Evolution Agent credential: {self.api_key_env}")
        endpoints = [
            {
                "endpoint": "primary",
                "model": self.model,
                "base_url": self.base_url,
                "api_key": primary_key,
                "retries": self.request_retries,
            },
            {
                "endpoint": "fallback",
                "model": self.fallback_model,
                "base_url": self.fallback_base_url,
                "api_key": self.fallback_api_key,
                "retries": self.fallback_request_retries,
            },
        ]
        errors: list[dict[str, Any]] = []
        for endpoint in endpoints:
            if not endpoint["api_key"]:
                errors.append(
                    {
                        "endpoint": endpoint["endpoint"],
                        "model": endpoint["model"],
                        "base_url": endpoint["base_url"],
                        "ok": False,
                        "error_type": "MissingAPIKey",
                        "error": "empty API key",
                    }
                )
                continue
            for attempt_index in range(int(endpoint["retries"]) + 1):
                payload = dict(request)
                payload["model"] = endpoint["model"]
                extra_body = _thinking_disabled_extra_body(str(endpoint["model"]), str(endpoint["base_url"]))
                if extra_body is None:
                    payload.pop("extra_body", None)
                else:
                    payload["extra_body"] = extra_body
                try:
                    from runtime_endpoint_transport import endpoint_transport_options

                    with endpoint_transport_options("evolution", endpoint["endpoint"], self.timeout_seconds) as options:
                        client = OpenAI(
                            base_url=str(endpoint["base_url"]),
                            api_key=str(endpoint["api_key"]),
                            timeout=self.timeout_seconds,
                            max_retries=0,
                            **options,
                        )
                        response = client.chat.completions.create(**payload)
                    self.last_request_audit.append(
                        {
                            "endpoint": endpoint["endpoint"],
                            "model": endpoint["model"],
                            "base_url": endpoint["base_url"],
                            "attempt": attempt_index + 1,
                            "ok": True,
                        }
                    )
                    return response
                except Exception as exc:  # noqa: BLE001 - failures are audited and retried.
                    record = {
                        "endpoint": endpoint["endpoint"],
                        "model": endpoint["model"],
                        "base_url": endpoint["base_url"],
                        "attempt": attempt_index + 1,
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                    }
                    self.last_request_audit.append(record)
                    errors.append(record)
                    if attempt_index < int(endpoint["retries"]) and self.retry_sleep_seconds:
                        time.sleep(self.retry_sleep_seconds * (attempt_index + 1))
        raise RuntimeError(
            "Evolution Agent request failed on primary and fallback endpoints: "
            + json.dumps(errors, ensure_ascii=False)
        )

    def propose(
        self,
        *,
        graph: ProcessorGraph,
        evidence: dict[str, Any],
        workspace: EvolutionWorkspace | None = None,
    ) -> dict[str, Any]:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"Missing Evolution Agent credential: {self.api_key_env}")
        model_evidence, trace_is_direct = _model_evidence_for_request(
            evidence=evidence,
            workspace=workspace,
        )
        current_artifacts = _artifact_bundle_for_request(workspace)
        workspace_brief = _workspace_brief(
            workspace,
            full_public_trace_in_initial_evidence=trace_is_direct,
        )
        prior_patch_history = _prior_artifact_patch_history_for_request(workspace)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": evolution_agent_system_prompt(self.policy),
            },
            {
                "role": "user",
                # Markdown prompt with machine-readable JSON blocks.
                # Small traces remain direct. Oversized traces remain exact in
                # the workspace and are available record-by-record, rather than
                # overflowing provider context or being summarized.
                "content": evolution_agent_user_prompt(
                    evidence=model_evidence,
                    current_artifacts=current_artifacts,
                    workspace=workspace_brief,
                    prior_artifact_patch_history=prior_patch_history,
                ),
            },
        ]
        self.last_session = copy.deepcopy(messages)
        self.last_tool_audit = []
        self.last_request_audit = []
        self.last_tool_rounds = 0
        workspace_tools = (
            EvolutionWorkspaceTools(
                workspace=workspace,
                graph=graph,
                policy=self.policy,
                evidence=evidence,
                evidence_trace_hashes=tuple(str(item) for item in evidence.get("trace_hashes", ()) if item),
            )
            if workspace is not None
            else None
        )
        tools = EvolutionWorkspaceTools.definitions() if workspace_tools is not None else None
        proposal: dict[str, Any] | None = None
        reserved_tool_rounds = 2 if self.max_tool_rounds >= 3 else 0
        reservation_notice_sent = False
        # The budget counts completed model-tool round trips. Once it is
        # exhausted we make one final model request *without* tool schemas,
        # forcing a direct JSON proposal rather than dropping a last attempted
        # write_candidate call at the boundary.
        for _turn in range(self.max_tool_rounds + 1):
            request: dict[str, Any] = {
                "model": self.model,
                "temperature": 0.0,
                "max_tokens": self.max_tokens,
                "messages": messages,
            }
            extra_body = _thinking_disabled_extra_body(self.model, self.base_url)
            if extra_body is not None:
                request["extra_body"] = extra_body
            tools_enabled = tools is not None and self.last_tool_rounds < self.max_tool_rounds
            if (
                tools_enabled
                and reserved_tool_rounds
                and not reservation_notice_sent
                and self.last_tool_rounds >= self.max_tool_rounds - reserved_tool_rounds
            ):
                reservation = {
                    "role": "user",
                    "content": (
                        "Only the final two workspace tool rounds remain. Stop reading additional files now. "
                        "If a non-empty artifact update is justified, use the next tool call to write the complete patch "
                        "to candidate/artifact_patch.json and the last one to call validate_candidate. If no edit is "
                        "justified, return the final JSON proposal now."
                    ),
                }
                messages.append(reservation)
                self.last_session.append(copy.deepcopy(reservation))
                request["messages"] = messages
                reservation_notice_sent = True
            if tools_enabled:
                request["tools"] = tools
                request["parallel_tool_calls"] = False
            elif workspace_tools is not None:
                finalization = {
                    "role": "user",
                    "content": (
                        "The Evolution workspace tool budget is exhausted. Use the complete source-blind evidence "
                        "you already read. Return the final JSON proposal now; do not call tools."
                    ),
                }
                messages.append(finalization)
                self.last_session.append(copy.deepcopy(finalization))
                request["messages"] = messages
            response = self._chat_completion_with_retry_and_fallback(request)
            message = response.choices[0].message
            tool_calls = list(message.tool_calls or [])
            assistant_message = _assistant_message_for_history(message)
            messages.append(assistant_message)
            self.last_session.append(copy.deepcopy(assistant_message))
            if not tool_calls:
                if workspace_tools is not None:
                    proposal = workspace_tools.load_candidate_patch()
                if proposal is None:
                    try:
                        proposal = _parse_model_json(message.content or "")
                    except Exception as exc:
                        repair = {
                            "role": "user",
                            "content": (
                                "Your previous response was not valid JSON for the fixed artifact patch contract: "
                                f"{type(exc).__name__}: {exc}. Return exactly one JSON object now, with keys "
                                "reason, changes, and file_updates. If no reusable update is justified, return "
                                '{"reason":"No reusable artifact change is justified from this episode.","changes":[],"file_updates":[]}. '
                                "Do not call tools."
                            ),
                        }
                        messages.append(repair)
                        self.last_session.append(copy.deepcopy(repair))
                        repair_request = dict(
                            model=self.model,
                            temperature=0.0,
                            max_tokens=self.max_tokens,
                            messages=messages,
                            **({"extra_body": extra_body} if (extra_body := _thinking_disabled_extra_body(self.model, self.base_url)) is not None else {}),
                        )
                        response = self._chat_completion_with_retry_and_fallback(repair_request)
                        message = response.choices[0].message
                        assistant_message = _assistant_message_for_history(message)
                        messages.append(assistant_message)
                        self.last_session.append(copy.deepcopy(assistant_message))
                        if message.tool_calls:
                            raise RuntimeError("Evolution Agent requested tools during no-tool JSON repair")
                        if workspace_tools is not None:
                            proposal = workspace_tools.load_candidate_patch()
                        if proposal is None:
                            proposal = _parse_model_json(message.content or "")
                break
            if workspace_tools is None:
                raise RuntimeError("Evolution model requested tools without an Evolution workspace")
            if not tools_enabled:
                raise RuntimeError("Evolution Agent requested a tool after the final no-tool proposal request")
            self.last_tool_rounds += 1
            for call in tool_calls:
                tool_name = call.function.name
                raw_arguments = call.function.arguments or "{}"
                arguments: dict[str, Any] | None = None
                argument_error: dict[str, str] | None = None
                started = time.perf_counter()
                try:
                    parsed_arguments = json.loads(raw_arguments)
                    if not isinstance(parsed_arguments, dict):
                        raise ValueError("Tool arguments must decode to an object")
                    arguments = parsed_arguments
                except Exception as exc:
                    argument_error = {"error_type": type(exc).__name__, "error": str(exc)}
                    result = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
                else:
                    result = workspace_tools.execute(tool_name, arguments)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": tool_name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
                self.last_tool_audit.append(
                    _tool_call_audit_record(
                        sequence=len(self.last_tool_audit) + 1,
                        tool_round=self.last_tool_rounds,
                        tool_call_id=str(call.id),
                        tool_name=tool_name,
                        raw_arguments=raw_arguments,
                        arguments=arguments,
                        argument_error=argument_error,
                        result=result,
                        tool_message=tool_message,
                        elapsed_seconds=time.perf_counter() - started,
                        workspace_tools=workspace_tools,
                    )
                )
                messages.append(tool_message)
                self.last_session.append(copy.deepcopy(tool_message))
        if proposal is None:
            raise RuntimeError("Evolution Agent returned no proposal")
        from .artifact_patcher import normalize_artifact_patch, validate_artifact_patch

        proposal = normalize_artifact_patch(proposal)
        validation_error = None
        try:
            validate_artifact_patch(proposal)
        except Exception as exc:
            validation_error = str(exc)
        if validation_error is not None:
            repair = {
                "role": "user",
                "content": (
                    "Your previous proposal failed artifact patch validation: "
                    f"{validation_error}. Return one complete corrected JSON proposal now; do not call tools. "
                    "The corrected proposal must contain changes and file_updates only, and may update only the fixed "
                    "Prompt, Memory, Skill, Permission, and Guard artifacts."
                ),
            }
            messages.append(repair)
            self.last_session.append(copy.deepcopy(repair))
            repair_request = dict(
                model=self.model,
                temperature=0.0,
                max_tokens=self.max_tokens,
                messages=messages,
                **({"extra_body": extra_body} if (extra_body := _thinking_disabled_extra_body(self.model, self.base_url)) is not None else {}),
            )
            response = self._chat_completion_with_retry_and_fallback(repair_request)
            message = response.choices[0].message
            assistant_message = _assistant_message_for_history(message)
            messages.append(assistant_message)
            self.last_session.append(copy.deepcopy(assistant_message))
            if message.tool_calls:
                raise RuntimeError("Evolution Agent requested tools during the required no-tool proposal repair")
            proposal = normalize_artifact_patch(_parse_model_json(message.content or ""))
            try:
                validate_artifact_patch(proposal)
            except Exception as exc:
                raise ValueError(
                    "Evolution Agent artifact patch remains structurally invalid after one repair: "
                    f"{exc}"
                ) from exc
        self.last_parse_repaired = bool(proposal.pop("_parser_repaired", False))
        if self.last_parse_repaired:
            # This travels with the persisted proposal, but candidate parsing
            # itself remains driven exclusively by the model's reason/edits.
            proposal["_parser_repaired"] = True
        return proposal


def _tool_call_audit_record(
    *,
    sequence: int,
    tool_round: int,
    tool_call_id: str,
    tool_name: str,
    raw_arguments: str,
    arguments: dict[str, Any] | None,
    argument_error: dict[str, str] | None,
    result: dict[str, Any],
    tool_message: dict[str, Any],
    elapsed_seconds: float,
    workspace_tools: EvolutionWorkspaceTools,
) -> dict[str, Any]:
    """Capture the exact tool result returned to the Evolution Agent.

    ``last_session`` remains the protocol-level transcript. This companion
    record is intentionally queryable: it establishes whether a particular
    source file (including generated Processor source) was returned by a
    workspace call, rather than inferring that fact from a final proposal.
    """

    payload = str(tool_message["content"])
    review_after = workspace_tools.source_review_status()
    received_path = result.get("path") if isinstance(result, dict) else None
    required_paths_received: list[str] = []
    if (
        tool_name == "read_workspace_file"
        and result.get("ok") is True
        and isinstance(received_path, str)
        and received_path in set(review_after["required_paths"])
    ):
        required_paths_received = [received_path]
    elif tool_name == "read_source_review_bundle" and result.get("ok") is True:
        files = result.get("files")
        if isinstance(files, list):
            required = set(review_after["required_paths"])
            required_paths_received = [
                item["path"]
                for item in files
                if isinstance(item, dict) and isinstance(item.get("path"), str) and item["path"] in required
            ]
    return {
        "schema_version": 1,
        "sequence": sequence,
        "tool_round": tool_round,
        "request": {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "raw_arguments": raw_arguments,
            "arguments": copy.deepcopy(arguments),
            "argument_error": copy.deepcopy(argument_error),
        },
        # This is the complete tool message appended to the next model request.
        # It intentionally includes returned source/trace content rather than a
        # lossy summary so receipt is independently auditable.
        "model_received": {
            "complete": True,
            "tool_message": copy.deepcopy(tool_message),
            "content_bytes": len(payload.encode("utf-8")),
            "content_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        },
        "source_review_after_call": {
            **review_after,
            "required_paths_received_by_this_call": required_paths_received,
        },
        "elapsed_seconds": round(elapsed_seconds, 6),
    }


class GeneratedProcessor(Processor):
    """Adapter for generated class processors.

    A class-style source is imported once when a frozen graph becomes a
    per-episode Harness. Its instance therefore keeps task-local state across
    all configured hooks. The graph's ``ProcessorSpec`` stays authoritative
    for placement and ordering, rather than trusting generated module fields.
    """

    def __init__(self, node: ProcessorNode) -> None:
        if node.implementation != "generated" or node.source is None or node.exported_class is None:
            raise ValueError("GeneratedProcessor needs a generated class source")
        validate_generated_source(node.source, exported_class=node.exported_class)
        self.node = node
        self.spec = ProcessorSpec(
            name=node.processor_id,
            hooks=frozenset(node.hooks),
            order=node.order,
            reads=frozenset({Resource.PROMPT, Resource.MEMORY, Resource.TOOL, Resource.CONTEXT, Resource.ACTION, Resource.SAFETY_OBSERVATION, Resource.EVOLUTION}),
            writes=frozenset({Resource.PROMPT, Resource.CONTEXT, Resource.EVOLUTION, Resource.ACTION, Resource.TOOL}),
            after=frozenset(node.after),
        )
        cls = _load_generated_processor_class(node.source, node.exported_class)
        try:
            instance = cls(**copy.deepcopy(node.parameters))
        except Exception as exc:
            raise ValueError(
                f"Generated Processor {node.processor_id} could not be instantiated: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(instance, MultiHookProcessor):
            raise ValueError(f"Generated Processor {node.processor_id} must subclass MultiHookProcessor")
        self._class_processor = instance
        instance.processor_id = node.processor_id
        instance.parameters = copy.deepcopy(node.parameters)

    def process(self, event: HarnessEvent) -> Any:
        return self._class_processor.process(event)


def initial_processor_graph(
    *,
    version: str = "H0",
    memory_enabled: bool = True,
    memory_limit: int = 3,
    guard_enabled: bool = True,
    guard_preflight_tool: bool = True,
    guard_action_mode: str = "advisory",
    permission_enabled: bool = True,
    permission_limit: int = 3,
    skill_enabled: bool = True,
    skill_auto_inject: bool = True,
    skill_load_tool: bool = True,
    skill_top_k: int = 3,
    skill_max_active: int = 8,
) -> ProcessorGraph:
    """Build the initial processor topology around the configured Guard."""

    if skill_max_active < 1:
        raise ValueError("skill_max_active must be at least 1")
    nodes = [
        ProcessorNode("memory_retriever", "MemoryRetriever", "builtin", (Hook.TASK_START,), 20, {"enabled": memory_enabled, "limit": memory_limit}),
        ProcessorNode("skill_catalog", "SkillCatalogProcessor", "builtin", (Hook.TASK_START,), 25, {"enabled": skill_enabled, "max_active": skill_max_active}),
        ProcessorNode("skill_runtime", "SkillRuntimeProcessor", "builtin", (Hook.TASK_START,), 26, {"enabled": skill_enabled}, after=("skill_catalog",)),
        ProcessorNode("permission_experience_injector", "PermissionExperienceInjector", "builtin", (Hook.TASK_START,), 27, {"enabled": permission_enabled, "limit": permission_limit}, after=("skill_runtime",)),
        ProcessorNode(
            "prompt_formatter",
            "PromptFormatter",
            "builtin",
            (Hook.TASK_START,),
            30,
            after=("permission_experience_injector",),
        ),
        ProcessorNode("progressive_skill_loader", "ProgressiveSkillLoader", "builtin", (Hook.BEFORE_MODEL,), 30, {"enabled": skill_enabled and skill_auto_inject, "top_k": skill_top_k}),
        ProcessorNode("skill_tool_provider", "SkillToolProvider", "builtin", (Hook.BEFORE_MODEL, Hook.BEFORE_TOOL), 40, {"enabled": skill_enabled, "expose_tool": skill_load_tool}),
        ProcessorNode("evolution_signal_detector", "EvolutionSignalDetector", "builtin", (Hook.AFTER_TOOL, Hook.TASK_END), 50),
        ProcessorNode("memory_writer", "MemoryWriter", "builtin", (Hook.TASK_END,), 70),
        ProcessorNode("trace_recorder", "TraceRecorder", "builtin", tuple(Hook), 1000),
    ]
    if guard_enabled:
        guard_mode = guard_action_mode
        nodes.append(ProcessorNode("guard_input_inspector", "GuardInputInspector", "builtin", (Hook.BEFORE_MODEL,), 44))
        if guard_preflight_tool:
            nodes.append(ProcessorNode("guard_preflight_tool_provider", "GuardPreflightToolProvider", "builtin", (Hook.BEFORE_MODEL, Hook.BEFORE_TOOL), 45))
        nodes.append(ProcessorNode("guard_action_gate", "GuardActionGate", "builtin", (Hook.BEFORE_TOOL,), 50, {"mode": guard_mode}))
        nodes.append(ProcessorNode("guard_output_inspector", "GuardOutputInspector", "builtin", (Hook.AFTER_TOOL,), 45))
    graph = ProcessorGraph(version=version, nodes=tuple(nodes))
    validate_graph(graph)
    return graph


def advance_graph_version(graph: ProcessorGraph, *, version: str) -> ProcessorGraph:
    successor = replace(graph, version=version)
    validate_graph(successor)
    return successor


def validate_graph(graph: ProcessorGraph) -> None:
    if graph.schema_version != GRAPH_SCHEMA_VERSION or not graph.version:
        raise ValueError("Invalid ProcessorGraph header")
    ids = [node.processor_id for node in graph.nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate processor_id in graph")
    known_ids = set(ids)
    for node in graph.nodes:
        if (
            not _identifier(node.processor_id)
            or not node.processor_name
            or not node.hooks
            or len(node.hooks) != len(set(node.hooks))
            or len(node.after) != len(set(node.after))
            or any(not _identifier(item) for item in node.after)
        ):
            raise ValueError(f"Invalid Processor node: {node.processor_id!r}")
        if not isinstance(node.order, int):
            raise ValueError(f"Processor order must be an integer: {node.processor_id}")
        unknown_after = sorted(set(node.after) - known_ids)
        if unknown_after:
            raise ValueError(
                f"Processor {node.processor_id} declares unknown after dependencies: {unknown_after}"
            )
        if node.implementation == "builtin":
            if node.processor_name not in _BUILTINS or node.source is not None or node.exported_class is not None:
                raise ValueError(f"Invalid builtin Processor: {node.processor_name}")
        elif node.implementation == "generated":
            if node.source is None or node.exported_class is None:
                raise ValueError(f"Generated Processor needs a class source: {node.processor_id}")
            validate_generated_source(node.source, exported_class=node.exported_class)
        else:
            raise ValueError(f"Unsupported Processor implementation: {node.implementation}")


def build_harness_from_graph(
    graph: ProcessorGraph,
    *,
    guard: Guard,
    trace_store: TraceStore,
    memory_store: MemoryStore,
    skill_registry: SkillRegistry | None = None,
    permission_store: PermissionExperienceStore | None = None,
    prompt_addendum_template: str | None = None,
    guard_policy: dict[str, Any] | None = None,
) -> "SafetyHarness":
    """Instantiate the frozen graph for exactly one episode."""

    from .runtime import HarnessBuilder, HarnessConfig

    validate_graph(graph)
    builder = HarnessBuilder(config=HarnessConfig(version=graph.version))
    registry = skill_registry or SkillRegistry()
    permission_experience_store = permission_store or InMemoryPermissionExperienceStore()
    for node in graph.nodes:
        builder.add(_instantiate(node, graph.version, guard, trace_store, memory_store, registry, permission_experience_store, prompt_addendum_template, guard_policy))
    return builder.build()


def evolution_agent_system_prompt(
    policy: EvolutionPolicy | None = None,
) -> str:
    """Protocol for fixed-artifact evolution."""

    del policy
    patch_template = json.dumps(artifact_patch_template(), ensure_ascii=False, indent=2, sort_keys=True)
    return render_prompt_template(
        "evolution_agent/system_prompt.md",
        PATCH_TEMPLATE_JSON=patch_template,
    )

def evolution_agent_user_prompt(
    *,
    evidence: dict[str, Any],
    current_artifacts: dict[str, Any] | None,
    workspace: dict[str, Any] | None,
    prior_artifact_patch_history: dict[str, Any] | None = None,
) -> str:
    """Render the Evolution Agent user message from a Markdown template."""

    task = evidence.get("task") if isinstance(evidence, dict) else None
    task_id = task.get("task_id") if isinstance(task, dict) else None
    contract = _artifact_update_contract_for_request(current_artifacts=current_artifacts)
    return render_prompt_template(
        "evolution_agent/user_prompt.md",
        TASK_ID=str(task_id or "unknown"),
        EVIDENCE_JSON=json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        CURRENT_ARTIFACTS_JSON=json.dumps(current_artifacts, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        ARTIFACT_UPDATE_CONTRACT_JSON=json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        PRIOR_ARTIFACT_PATCH_HISTORY_JSON=json.dumps(prior_artifact_patch_history or {"included": False}, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        WORKSPACE_JSON=json.dumps(workspace, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        PATCH_SCHEMA_JSON=json.dumps(artifact_patch_template(), ensure_ascii=False, indent=2, sort_keys=True),
    )


def _prior_artifact_patch_history_for_request(workspace: EvolutionWorkspace | None) -> dict[str, Any]:
    """Return source-blind prior artifact patch history for the user prompt."""

    if workspace is None:
        return {"included": False, "reason": "workspace_unavailable"}
    path = workspace.root / "history" / "artifact_patch_ledger.jsonl"
    rows: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return {
        "included": True,
        "delivery": "source_blind_prior_artifact_patch_ledger",
        "path_in_workspace": "history/artifact_patch_ledger.jsonl",
        "entry_count": len(rows),
        "entries": rows[-20:],
        "use": (
            "Use this to avoid repeated low-level edits, identify recurring mechanisms, "
            "and make future artifact changes more high-level and stable. Do not treat history as proof by itself; "
            "current official feedback remains the completed-episode ground truth."
        ),
    }


def _artifact_update_contract_for_request(*, current_artifacts: dict[str, Any] | None) -> dict[str, Any]:
    files = (current_artifacts or {}).get("files") if isinstance(current_artifacts, dict) else None
    skill_registry = None
    if isinstance(files, dict):
        registry = files.get("skills/registry.json")
        if isinstance(registry, dict):
            try:
                skill_registry = json.loads(str(registry.get("content") or "{}"))
            except json.JSONDecodeError:
                skill_registry = None
    active_skill_count = None
    skill_registry_version = None
    skill_registry_hash = None
    if isinstance(skill_registry, dict):
        skills = skill_registry.get("skills")
        if isinstance(skills, list):
            active_skill_count = sum(1 for item in skills if isinstance(item, dict) and item.get("status", "active") == "active")
        skill_registry_version = skill_registry.get("version")
        skill_registry_hash = skill_registry.get("registry_hash")
    allowed_paths = [
        "prompt/target_system_addendum.md",
        "memory/validated_experience.jsonl",
        "skills/registry.json",
        "skills/<skill_id>/SKILL.md",
        "permission/permission_experience.jsonl",
        "guard/guard_policy.json",
    ]
    runtime_effects = {
        "prompt": "prompt/target_system_addendum.md updates the next episode Target Agent system prompt addendum.",
        "memory": "memory/validated_experience.jsonl updates MemoryStore; later episodes see records only when retrieval selects them.",
        "skill": "skills/registry.json and SKILL.md update SkillRegistry for future episodes; catalog, auto injection, and LoadSkill use the successor registry.",
        "permission": "permission/permission_experience.jsonl updates PermissionExperienceStore; later episodes see lessons only when retrieval selects them.",
        "guard": "guard/guard_policy.json updates Target-visible Guard verdict interpretation text; it does not change the Guard model or create automatic blocking.",
    }
    return {
        "allowed_paths": allowed_paths,
        "allowed_modes": ["append_text", "replace_text", "append_jsonl", "update_jsonl", "replace_json", "json_patch"],
        "forbidden_changes": ["processor_graph", "hooks", "runtime_code", "tools", "evaluator", "dataset", "model_configuration", "benchmark_results"],
        "runtime_effects": runtime_effects,
        "skill_update_boundary": {
            "prefer_modify_existing_skill": True,
            "minimum_distinct_completed_episodes_for_new_skill": 3,
            "active_skill_count": active_skill_count,
            "current_registry_version": skill_registry_version,
            "current_registry_hash": skill_registry_hash,
        },
        "source_blind_contract": {
            "do_not_name_benchmark_or_source_adapter": True,
            "do_not_store_task_ids_exact_tool_names_entities_payloads_or_answers_in_reusable_artifacts": True,
            "official_feedback_is_completed_episode_ground_truth": True,
        },
    }


def validate_generated_source(source: str, *, exported_class: str) -> None:
    """Validate a generated Processor class without executing task data."""

    if not isinstance(source, str) or not source.strip() or len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("Invalid generated Processor source size")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"Generated Processor syntax error: {exc.msg}") from exc
    _validate_class_generated_source(tree, exported_class)


def _validate_class_generated_source(tree: ast.Module, exported_class: str) -> None:
    """Validate the static shape of a six-hook processor class."""

    target = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == exported_class),
        None,
    )
    if target is None:
        raise ValueError(f"Generated Processor source does not define exported_class {exported_class!r}")
    if not any(_ast_name(base).endswith("MultiHookProcessor") for base in target.bases):
        raise ValueError("Generated Processor exported class must inherit MultiHookProcessor")
    declared_hooks: list[str] = []
    for item in target.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) or not item.name.startswith("on_"):
            continue
        hook_name = item.name.removeprefix("on_")
        if hook_name not in {hook.value for hook in Hook}:
            raise ValueError(f"Generated Processor uses unsupported hook: {item.name}")
        if not isinstance(item, ast.AsyncFunctionDef):
            raise ValueError(f"Generated Processor hook must be async: {item.name}")
        if not any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in ast.walk(item)):
            raise ValueError(f"Generated Processor hook must yield one or more events: {item.name}")
        declared_hooks.append(hook_name)
    if not declared_hooks:
        raise ValueError("Generated Processor must implement at least one on_<hook> method")
    if any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "process" for item in target.body):
        raise ValueError("Generated Processor must use MultiHookProcessor dispatch instead of defining process")


def _ast_name(value: ast.expr) -> str:
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    return ""


def _load_generated_processor_class(source: str, exported_class: str) -> type[MultiHookProcessor]:
    """Load one validated generated class into this per-episode Harness.

    The dynamically created module is intentionally unique per construction;
    instance state is never shared across episodes. Long-lived learning must
    use the managed Harness state/memory path, not module globals.
    """

    validate_generated_source(source, exported_class=exported_class)
    module_name = f"_safety_harness_generated_{uuid.uuid4().hex}"
    module = types.ModuleType(module_name)
    module.__file__ = f"<{module_name}>"
    try:
        exec(compile(source, module.__file__, "exec"), module.__dict__, module.__dict__)
    except Exception as exc:
        raise ValueError(
            f"Generated Processor module failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    candidate = getattr(module, exported_class, None)
    if not isinstance(candidate, type) or not issubclass(candidate, MultiHookProcessor):
        raise ValueError(f"Generated Processor export {exported_class!r} is not a MultiHookProcessor class")
    return candidate


def _instantiate(
    node: ProcessorNode,
    version: str,
    guard: Guard,
    trace_store: TraceStore,
    memory_store: MemoryStore,
    skill_registry: SkillRegistry,
    permission_store: PermissionExperienceStore,
    prompt_addendum_template: str | None = None,
    guard_policy: dict[str, Any] | None = None,
) -> Processor:
    if node.implementation == "generated":
        return GeneratedProcessor(node)
    params = node.parameters
    if node.processor_name == "MemoryRetriever":
        processor: Processor = MemoryRetriever(memory_store, enabled=bool(params.get("enabled", False)), limit=int(params.get("limit", 3)))
    elif node.processor_name == "PromptFormatter":
        processor = PromptFormatter(addendum_template=prompt_addendum_template, guard_policy=guard_policy)
    elif node.processor_name == "SkillCatalogProcessor":
        processor = SkillCatalogProcessor(skill_registry, enabled=bool(params.get("enabled", True)), max_active=int(params.get("max_active", 8)))
    elif node.processor_name == "SkillRuntimeProcessor":
        processor = SkillRuntimeProcessor(skill_registry, enabled=bool(params.get("enabled", True)))
    elif node.processor_name == "PermissionExperienceInjector":
        processor = PermissionExperienceInjector(permission_store, enabled=bool(params.get("enabled", True)), limit=int(params.get("limit", 3)))
    elif node.processor_name == "ProgressiveSkillLoader":
        processor = ProgressiveSkillLoader(skill_registry, enabled=bool(params.get("enabled", True)), top_k=int(params.get("top_k", 3)))
    elif node.processor_name == "SkillToolProvider":
        processor = SkillToolProvider(skill_registry, enabled=bool(params.get("enabled", True)), expose_tool=bool(params.get("expose_tool", True)))
    elif node.processor_name == "GuardPreflightToolProvider":
        processor = GuardPreflightToolProvider(guard, enabled=bool(params.get("enabled", True)), expose_tool=bool(params.get("expose_tool", True)))
    elif node.processor_name == "GuardInputInspector":
        processor = GuardInputInspector(guard, enabled=bool(params.get("enabled", True)))
    elif node.processor_name == "GuardActionGate":
        processor = GuardActionGate(guard, mode=str(params.get("mode", "advisory")))
    elif node.processor_name == "GuardOutputInspector":
        processor = GuardOutputInspector(guard, enabled=bool(params.get("enabled", True)))
    elif node.processor_name == "EvolutionSignalDetector":
        processor = EvolutionSignalDetector(harness_version=version)
    elif node.processor_name == "MemoryWriter":
        processor = MemoryWriter(memory_store)
    elif node.processor_name == "GuardCaller":
        processor = GuardCaller(guard)
    elif node.processor_name == "TraceRecorder":
        processor = TraceRecorder(trace_store)
    else:  # validate_graph excludes this branch.
        raise ValueError(node.processor_name)
    processor.spec = replace(
        processor.spec,
        name=node.processor_id,
        hooks=frozenset(node.hooks),
        order=node.order,
        after=frozenset(node.after),
    )
    return processor


def _parse_model_json(content: str) -> dict[str, Any]:
    value = content.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) < 3 or not lines[-1].strip().startswith("```"):
            raise ValueError("Invalid fenced Evolution Agent output")
        value = "\n".join(lines[1:-1])
    repaired = False
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as strict_error:
        # Compatible endpoints occasionally return a JSON-like object with an
        # unescaped newline in generated source or a missing final quote. A
        # repaired object is still subject to the normal graph schema, source
        # sandbox, and candidate smoke validation; malformed content cannot be
        # deployed merely because it was repairable.
        try:
            # The standard decoder can safely preserve a literal newline in a
            # generated source string without any third-party dependency.
            parsed = json.loads(value, strict=False)
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json

                parsed = repair_json(value, return_objects=True)
            except Exception as repair_error:
                parsed = _embedded_evolution_json(value)
                if parsed is None:
                    raise ValueError(f"Invalid Evolution Agent JSON: {strict_error}") from repair_error
        # ``json_repair`` can return a syntactically valid but unrelated
        # scalar/object when prose and provider control tags surround the
        # proposal. Prefer a balanced embedded object only when it contains
        # the two required Evolution fields.
        if not _looks_like_evolution_json(parsed):
            embedded = _embedded_evolution_json(value)
            if embedded is not None:
                parsed = embedded
        if not isinstance(parsed, dict):
            raise ValueError(f"Invalid Evolution Agent JSON: {strict_error}")
        repaired = True
    if not isinstance(parsed, dict):
        raise ValueError("Evolution Agent output must be a JSON object")
    if repaired:
        parsed["_parser_repaired"] = True
    return parsed


def _looks_like_evolution_json(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and (
            isinstance(value.get("edits"), list)
            or isinstance(value.get("file_updates"), list)
            or isinstance(value.get("changes"), list)
        )
    )


def _embedded_evolution_json(value: str) -> dict[str, Any] | None:
    """Recover one proposal object from provider template leakage.

    Some OpenAI-compatible endpoints return an explanation followed by a JSON
    proposal and proprietary ``DSML`` closing tags even when the system prompt
    asks for JSON only. We accept only a balanced object that has the required
    Evolution proposal keys; arbitrary embedded JSON is not a candidate.
    """

    candidates: list[dict[str, Any]] = []
    for start, character in enumerate(value):
        if character != "{":
            continue
        end = _balanced_json_object_end(value, start)
        if end is None:
            continue
        fragment = value[start:end]
        try:
            parsed = json.loads(fragment)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(fragment, strict=False)
            except json.JSONDecodeError:
                continue
        if isinstance(parsed, dict) and isinstance(parsed.get("reason"), str) and isinstance(parsed.get("edits"), list):
            candidates.append(parsed)
    return candidates[-1] if candidates else None


def _balanced_json_object_end(value: str, start: int) -> int | None:
    """Return the exclusive end of a quote-aware balanced JSON object."""

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(value)):
        character = value[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index + 1
            if depth < 0:
                return None
    return None


def _model_evidence_for_request(
    *, evidence: dict[str, Any], workspace: EvolutionWorkspace | None
) -> tuple[dict[str, Any], bool]:
    """Prefer direct full evidence; use indexed trace access only as fallback.

    The workspace always persists the complete source-blind public trace. The
    Evolution Agent should normally receive the just-completed episode trace in
    the initial request, so workspace trace readers are reserved for genuinely
    oversized traces rather than routine evidence gathering.
    """

    serialized = json.dumps(evidence, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    if workspace is None or len(serialized) <= MAX_DIRECT_EVOLUTION_EVIDENCE_BYTES:
        return copy.deepcopy(evidence), True
    trace = evidence.get("public_trace")
    if not isinstance(trace, list):
        return copy.deepcopy(evidence), True
    model_evidence = copy.deepcopy(evidence)
    model_evidence["public_trace"] = {
        "delivery": "workspace_indexed_exact_untruncated",
        "full_trace_path": workspace.relative(workspace.trace_path),
        "trace_index_path": workspace.relative(workspace.trace_index_path),
        "record_count": len(trace),
        "raw_trace_sha256": hashlib.sha256(
            json.dumps(trace, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "raw_trace_bytes": workspace.trace_path.stat().st_size,
        "access": "Read the exact index, then call read_trace_record(index) for an untruncated original record.",
    }
    model_evidence["public_trace_delivery"] = "workspace_indexed_exact_untruncated"
    model_evidence["raw_trace_integrity"] = (
        "The complete append-only source-blind public trace remains unchanged in the workspace. "
        "This request uses exact record access only because direct delivery exceeds the provider context budget."
    )
    return model_evidence, False


def _artifact_bundle_for_request(workspace: EvolutionWorkspace | None) -> dict[str, Any] | None:
    """Return complete current artifact contents in the initial request."""

    if workspace is None:
        return None
    artifact_root = workspace.root / "artifacts"
    if not artifact_root.is_dir():
        return {"included": False, "reason": "artifact_snapshot_unavailable"}
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in artifact_root.rglob("*") if item.is_file()):
        rel = str(path.relative_to(artifact_root))
        content = path.read_text(encoding="utf-8")
        files[rel] = {
            "bytes": len(content.encode("utf-8")),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "content": content,
        }
    return {
        "included": True,
        "delivery": "complete_direct_initial_message",
        "root_in_workspace": "artifacts/",
        "files": files,
    }

def _workspace_brief(
    workspace: EvolutionWorkspace | None,
    *,
    full_public_trace_in_initial_evidence: bool,
) -> dict[str, Any] | None:
    """Describe the optional file-tool workspace and trace delivery mode."""

    if workspace is None:
        return None
    return {
        "root": str(workspace.root),
        "manifest": "workspace_manifest.json",
        "read_tools": ["read_workspace_file", "read_source_review_bundle", "read_trace_record", "read_episode_trace_record", "glob_workspace", "grep_workspace"],
        "batch_trace_tool": "read_episode_trace_record",
        "write_tool": "write_candidate_file",
        "write_root": "candidate/",
        "proposal_path": "candidate/artifact_patch.json",
        "validation_tool": "validate_candidate",
        "source_review_required_for_nonempty_candidate": False,
        "active_artifacts": "artifacts/",
        "full_public_trace_path": "episode/public_trace.json",
        "full_public_trace_index_path": "episode/public_trace_index.json",
        "full_public_trace_also_in_initial_evidence": full_public_trace_in_initial_evidence,
        "oversize_trace_access": "Use trace reader tools only when initial evidence explicitly says the trace is workspace-indexed or a needed record is missing; otherwise the completed episode trace is already in the initial message.",
    }


def _assistant_message_for_history(message: Any) -> dict[str, Any]:
    """Convert a compatible OpenAI response message into the next request."""

    tool_calls = []
    for call in message.tool_calls or []:
        tool_calls.append(
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments or "{}",
                },
            }
        )
    payload: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if tool_calls:
        payload["tool_calls"] = tool_calls
    return payload


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _identifier(value: str) -> bool:
    return value.isidentifier() and not value.startswith("_")


_BUILTINS = frozenset({
    "MemoryRetriever", "PromptFormatter", "EvolutionSignalDetector", "MemoryWriter", "GuardCaller", "GuardInputInspector", "GuardActionGate", "GuardOutputInspector", "TraceRecorder",
    "SkillCatalogProcessor", "SkillRuntimeProcessor", "ProgressiveSkillLoader", "SkillToolProvider", "GuardPreflightToolProvider",
    "PermissionExperienceInjector",
})
