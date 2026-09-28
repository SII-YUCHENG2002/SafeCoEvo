"""Read-only, auditable workspace for one online Evolution Agent episode.

The Evolution Agent must be able to inspect the complete public evidence and
the actual Harness implementation before it proposes artifact changes. This
module materializes those inputs as a per-episode snapshot, then exposes a
small file-tool surface that can read only that snapshot and write only its
candidate directory. The controller remains the sole component that can
validate and deploy an artifact patch.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .artifact_patcher import ArtifactState
    from .evolution_graph import EvolutionPolicy, ProcessorGraph
    from .memory import MemoryStore
    from .permission import PermissionExperienceStore
    from .skills import SkillRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_SCHEMA_VERSION = 4
MAX_CANDIDATE_FILE_BYTES = 512_000
MAX_GREP_MATCHES = 40
MAX_GREP_CONTENT_CHARS = 2_000
MAX_TOP_LEVEL_NESTED_TRACE_READ_BYTES = 256_000
SOURCE_REVIEW_MANIFEST = "harness/source_review_manifest.json"
SOURCE_REVIEW_STATUS = "harness/source_review_status.json"


# These are the source-neutral implementation modules needed to understand the
# Processor interface and built-ins. Source executors are intentionally absent:
# their benchmark identities and evaluator routing are coordinator-only.
# A snapshot prevents the Evolution Agent from reading mutable project files
# while a benchmark stream is in progress.
_PUBLIC_RUNTIME_FILES: tuple[tuple[str, Path], ...] = (
    ("harness_api/contracts.py", PROJECT_ROOT / "src/safecoevo_runtime/contracts.py"),
    ("harness_api/events.py", PROJECT_ROOT / "src/safecoevo_runtime/events.py"),
    ("harness_api/processor.py", PROJECT_ROOT / "src/safecoevo_runtime/processor.py"),
    ("harness_api/runtime.py", PROJECT_ROOT / "src/safecoevo_runtime/runtime.py"),
    ("harness_api/processors.py", PROJECT_ROOT / "src/safecoevo_runtime/processors.py"),
    ("harness_api/memory.py", PROJECT_ROOT / "src/safecoevo_runtime/memory.py"),
    ("harness_api/target_prompts.py", PROJECT_ROOT / "src/safecoevo_runtime/target_prompts.py"),
    ("harness_api/trace.py", PROJECT_ROOT / "src/safecoevo_runtime/trace.py"),
    ("harness_api/evolution_graph.py", PROJECT_ROOT / "src/safecoevo_runtime/evolution_graph.py"),
    ("harness_api/evolution_workspace.py", PROJECT_ROOT / "src/safecoevo_runtime/evolution_workspace.py"),
)

# This is the bounded implementation closure of the active Processor graph.
# It deliberately excludes source executors and evaluators, which are not
# needed to understand Processor behavior and would leak benchmark provenance.
_SOURCE_REVIEW_RUNTIME_PATHS = (
    "harness_api/contracts.py",
    "harness_api/events.py",
    "harness_api/processor.py",
    "harness_api/runtime.py",
    "harness_api/processors.py",
    "harness_api/memory.py",
    "harness_api/target_prompts.py",
    "harness_api/trace.py",
)


@dataclass(frozen=True)
class EvolutionWorkspace:
    """Immutable public inputs plus a single writable candidate directory."""

    root: Path
    candidate_root: Path
    evidence_path: Path
    trace_path: Path
    trace_index_path: Path
    manifest_path: Path

    def relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.root.resolve()))


def create_evolution_workspace(
    *,
    evolution_root: Path,
    task_id: str,
    graph: "ProcessorGraph",
    evidence: dict[str, Any],
    policy: "EvolutionPolicy",
    memory_store: "MemoryStore",
    skill_registry: "SkillRegistry | None" = None,
    permission_store: "PermissionExperienceStore | None" = None,
    artifact_state: "ArtifactState | None" = None,
    skill_review_window: int = 1,
) -> EvolutionWorkspace:
    """Create the read-only public snapshot available for one evolution step.

    ``evidence`` deliberately already contains the complete untruncated
    source-blind trace. It is persisted verbatim, alongside an exact-record index. The
    model adapter can use that index when direct delivery would exceed a
    provider context limit; no trace content is summarized or discarded.
    """

    root = evolution_root / "episodes" / task_id / "evolution_workspace"
    candidate_root = root / "candidate"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    candidate_root.mkdir(mode=0o700, parents=True, exist_ok=True)

    task = copy.deepcopy(evidence.get("task", {}))
    feedback = copy.deepcopy(evidence.get("official_feedback", {}))
    trace = copy.deepcopy(evidence.get("public_trace", []))
    _write_json(root / "episode" / "task.json", task)
    _write_json(root / "episode" / "official_feedback.json", feedback)
    trace_path = root / "episode" / "public_trace.json"
    trace_index_path = root / "episode" / "public_trace_index.json"
    _write_json(trace_path, trace)
    _write_json(trace_index_path, _trace_index(trace))
    evidence_path = root / "episode" / "evidence.json"
    _write_json(evidence_path, evidence)

    _write_json(root / "harness" / "active_graph.json", graph.to_dict())
    _write_json(root / "harness" / "policy.json", policy.to_dict())
    _write_json(root / "harness" / "candidate_schema.json", _candidate_schema())

    if artifact_state is not None:
        _write_text(root / "artifacts" / "prompt" / "target_system_addendum.md", artifact_state.prompt_addendum)
        _write_json(root / "artifacts" / "guard" / "guard_policy.json", artifact_state.guard_policy)
    _write_jsonl(
        root / "artifacts" / "memory" / "validated_experience.jsonl",
        [_jsonable(item) for item in getattr(memory_store, "items", ())],
    )
    _write_jsonl(
        root / "artifacts" / "permission" / "permission_experience.jsonl",
        [_jsonable(item) for item in getattr(permission_store, "items", ())] if permission_store is not None else [],
    )
    if skill_registry is not None:
        _write_json(root / "artifacts" / "skills" / "registry.json", skill_registry.to_dict())
        for skill in skill_registry.active:
            _write_text(root / "artifacts" / "skills" / skill.skill_id / "SKILL.md", skill.content)
    if skill_registry is not None:
        _write_json(root / "harness" / "skills" / "registry.json", skill_registry.to_dict())
        for skill in skill_registry.active:
            _write_text(root / "harness" / "skills" / "active" / skill.skill_id / "SKILL.md", skill.content)
    generated_sources: list[dict[str, Any]] = []
    for node in graph.nodes:
        if node.implementation == "generated" and node.source is not None:
            _write_text(root / "harness" / "active_generated_processors" / f"{node.processor_id}.py", node.source)
            generated_sources.append(
                {
                    "processor_id": node.processor_id,
                    "processor_name": node.processor_name,
                    "hooks": [hook.value for hook in node.hooks],
                    "source": node.source,
                    "exported_class": node.exported_class,
                }
            )
    # One bundle avoids making the source-review protocol consume a tool round
    # per generated Processor when a graph has accumulated many of them.
    _write_json(
        root / "harness" / "active_generated_processors.json",
        {"schema_version": 1, "processors": generated_sources},
    )

    source_manifest: list[dict[str, Any]] = []
    for relative, source in _PUBLIC_RUNTIME_FILES:
        if not source.is_file():
            # Optional adapters may be absent in a reduced checkout.  Their
            # absence is explicit in the manifest, never silently hidden.
            source_manifest.append({"path": relative, "available": False})
            continue
        destination = root / relative
        _copy_file(source, destination)
        source_manifest.append(
            {
                "path": relative,
                "available": True,
                "sha256": _sha256_file(destination),
            }
        )

    source_review_manifest = _source_review_manifest(root=root, graph=graph, skill_registry=skill_registry)
    _write_json(root / SOURCE_REVIEW_MANIFEST, source_review_manifest)
    _write_json(
        root / SOURCE_REVIEW_STATUS,
        {
            "schema_version": 1,
            "required_paths": source_review_manifest["required_paths"],
            "reviewed_paths": [],
            "complete": False,
        },
    )

    _snapshot_source_blind_history(
        evolution_root=evolution_root,
        task_id=task_id,
        destination=root / "history",
        review_window=skill_review_window,
    )
    memory_items = getattr(memory_store, "items", ())
    _write_json(root / "history" / "verified_memory.json", [_jsonable(item) for item in memory_items])

    _write_text(
        candidate_root / "README.md",
        "Write candidate-only artifacts here. The controller reads candidate/artifact_patch.json as a fixed-artifact patch. "
        "Do not write generated Processor code or new artifact categories. "
        "All other workspace paths are read-only to the Evolution Agent.\n",
    )
    _write_text(root / "README.md", _workspace_readme())
    manifest_path = root / "workspace_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "view_contract": "source_blind_evolution_workspace",
            "task_id": task_id,
            "public_trace_delivery": "complete_untruncated_direct_or_exact_record_indexed",
            "public_trace_path": "episode/public_trace.json",
            "public_trace_index_path": "episode/public_trace_index.json",
            "public_trace_record_access": "read_trace_record returns one exact untruncated record by index",
            "private_evaluator_included": False,
            "benchmark_provenance_included": False,
            "read_root": ".",
            "write_root": "candidate",
            "candidate_artifact_patch_path": "candidate/artifact_patch.json",
            "candidate_skill_manifest_path": "candidate/skills/manifest.json",
            "source_snapshot": source_manifest,
            "active_generated_processor_ids": [
                node.processor_id for node in graph.nodes if node.implementation == "generated"
            ],
            "active_artifacts": {
                "prompt": "artifacts/prompt/target_system_addendum.md",
                "memory": "artifacts/memory/validated_experience.jsonl",
                "skill_registry": "artifacts/skills/registry.json",
                "permission": "artifacts/permission/permission_experience.jsonl",
                "guard": "artifacts/guard/guard_policy.json",
            },
            "active_skill_registry": (
                {
                    "path": "harness/skills/registry.json",
                    "version": skill_registry.version,
                    "registry_hash": skill_registry.registry_hash,
                    "active_skill_ids": [skill.skill_id for skill in skill_registry.active],
                }
                if skill_registry is not None
                else None
            ),
            "source_review": {
                "required_for": "any_nonempty_candidate",
                "manifest_path": SOURCE_REVIEW_MANIFEST,
                "status_path": SOURCE_REVIEW_STATUS,
                "required_paths": source_review_manifest["required_paths"],
                "unavailable_paths": source_review_manifest["unavailable_paths"],
            },
            "history": {
                "prior_deployment_summary": "history/deployments.jsonl",
                "prior_artifact_patch_ledger": "history/artifact_patch_ledger.jsonl",
                "prior_graph_lineage": "history/graph_versions.json",
                "stream_feedback_summary": "history/episode_outcomes.jsonl",
                "verified_memory": "history/verified_memory.json",
                "review_window": "history/review_window/ contains exact source-blind public traces and released feedback for preceding episodes in the configured Skill review window.",
            },
            "integrity": {
                "evidence_sha256": _sha256_file(evidence_path),
                "active_graph_sha256": graph.graph_hash,
                "public_trace_sha256": _sha256_value(trace),
                "public_trace_bytes": trace_path.stat().st_size,
                "public_trace_record_count": len(trace),
            },
        },
    )
    return EvolutionWorkspace(
        root=root,
        candidate_root=candidate_root,
        evidence_path=evidence_path,
        trace_path=trace_path,
        trace_index_path=trace_index_path,
        manifest_path=manifest_path,
    )


def _source_review_manifest(
    *,
    root: Path,
    graph: "ProcessorGraph",
    skill_registry: "SkillRegistry | None",
) -> dict[str, Any]:
    """Describe the complete active Processor implementation closure.

    ``active_generated_processors.json`` contains every active generated
    source in a single exact file. Builtin Processor behavior is defined by
    the bounded source-neutral runtime files below. This makes review
    auditable without exposing source executors or requiring one tool round
    per generated Processor.
    """

    required_paths = [
        "harness/active_graph.json",
        "harness/active_generated_processors.json",
        *_SOURCE_REVIEW_RUNTIME_PATHS,
    ]
    if skill_registry is not None:
        required_paths.append("harness/skills/registry.json")
        required_paths.extend(
            f"harness/skills/active/{skill.skill_id}/SKILL.md"
            for skill in skill_registry.active
        )
    unavailable = [path for path in required_paths if not (root / path).is_file()]
    return {
        "schema_version": 1,
        "contract": "nonempty_candidate_requires_complete_active_processor_source_review",
        "required_paths": required_paths,
        "unavailable_paths": unavailable,
        "active_processor_ids": [node.processor_id for node in graph.nodes],
        "generated_processor_ids": [
            node.processor_id for node in graph.nodes if node.implementation == "generated"
        ],
        "notes": {
            "generated_sources": "harness/active_generated_processors.json contains every active generated Processor source.",
            "builtin_sources": "The required harness_api files contain the active builtin Processor implementations and their direct runtime behavior.",
            "skill_sources": "harness/skills/registry.json and harness/skills/active/ contain the complete active versioned Skill set.",
            "excluded": "Source executors, evaluator code, benchmark labels, and private oracle details are intentionally unavailable.",
        },
    }


class EvolutionWorkspaceTools:
    """Controlled tools used by the model-side Evolution Agent loop."""

    def __init__(
        self,
        *,
        workspace: EvolutionWorkspace,
        graph: "ProcessorGraph",
        policy: "EvolutionPolicy",
        evidence: dict[str, Any],
        evidence_trace_hashes: tuple[str, ...],
    ) -> None:
        self.workspace = workspace
        self.graph = graph
        self.policy = policy
        self.evidence = copy.deepcopy(evidence)
        self.evidence_trace_hashes = evidence_trace_hashes
        self._source_review_manifest = self._load_source_review_manifest()
        self._reviewed_source_paths = self._load_reviewed_source_paths()

    @staticmethod
    def definitions() -> list[dict[str, Any]]:
        """OpenAI function definitions; the paths are workspace-relative."""

        return [
            {
                "type": "function",
                "function": {
                    "name": "read_workspace_file",
                    "description": "Read an exact UTF-8 file from the read-only evolution workspace. No content is truncated by this tool.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string", "description": "Workspace-relative file path."}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_source_review_bundle",
                    "description": "Read the complete active-Processor source-review closure in one call. It returns every file required by harness/source_review_manifest.json with exact untruncated content, marks every required path reviewed, and never includes source executors or private evaluators.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_trace_record",
                    "description": "Read one exact, untruncated public trace record by its zero-based index. Inspect episode/public_trace_index.json first when the trace is workspace-indexed because the complete raw trace is intentionally not summarized.",
                    "parameters": {
                        "type": "object",
                        "properties": {"index": {"type": "integer", "minimum": 0}},
                        "required": ["index"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_episode_trace_record",
                    "description": "For batch evidence, read one exact trace event from one batch episode without loading the whole episode trace. Use this instead of read_trace_record when episode/public_trace_index.json shows nested episode traces.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "episode_index": {"type": "integer", "minimum": 1},
                            "record_index": {"type": "integer", "minimum": 0},
                        },
                        "required": ["episode_index", "record_index"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "glob_workspace",
                    "description": "List workspace-relative files matching a glob pattern, for example harness_api/*.py or history/episodes/**/*.json.",
                    "parameters": {
                        "type": "object",
                        "properties": {"pattern": {"type": "string"}},
                        "required": ["pattern"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "grep_workspace",
                    "description": "Search exact UTF-8 workspace files with a regular expression. It returns every matching line and never changes files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "path": {"type": "string", "description": "Optional workspace-relative file or directory; defaults to ."},
                        },
                        "required": ["pattern"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_candidate_file",
                    "description": "Write a UTF-8 candidate artifact under candidate/ only. Use candidate/artifact_patch.json for fixed artifact patches.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["path", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "validate_candidate",
                    "description": "Run structural validation for candidate/artifact_patch.json. It never deploys, edits runtime code, or replays a benchmark task.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string", "description": "Defaults to candidate/artifact_patch.json."}},
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run one controlled tool call and expose failures as model-visible data."""

        try:
            if name == "read_workspace_file":
                return self._read(arguments)
            if name == "read_source_review_bundle":
                return self._read_source_review_bundle(arguments)
            if name == "read_trace_record":
                return self._read_trace_record(arguments)
            if name == "read_episode_trace_record":
                return self._read_episode_trace_record(arguments)
            if name == "glob_workspace":
                return self._glob(arguments)
            if name == "grep_workspace":
                return self._grep(arguments)
            if name == "write_candidate_file":
                return self._write_candidate(arguments)
            if name == "validate_candidate":
                return self._validate_candidate(arguments)
            raise ValueError(f"Unsupported Evolution workspace tool: {name}")
        except Exception as exc:
            return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}

    def load_candidate_patch(self) -> dict[str, Any] | None:
        path = self.workspace.candidate_root / "artifact_patch.json"
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("candidate/artifact_patch.json must be a JSON object")
        return value

    def _read(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._read_path(_string(arguments, "path"))
        if not path.is_file():
            raise ValueError("read_workspace_file requires a regular file")
        content = path.read_text(encoding="utf-8")
        relative = self.workspace.relative(path)
        if relative in set(self._source_review_manifest["required_paths"]):
            self._reviewed_source_paths.add(relative)
            self._write_source_review_status()
        return {"ok": True, "path": relative, "bytes": len(content.encode("utf-8")), "content": content}

    def _read_source_review_bundle(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Return every required Processor source while recording each receipt.

        A non-empty update requires the entire implementation closure, not ten
        separate model turns. The returned records remain exact and the review
        receipt still names every path, so this is a latency optimization rather
        than a relaxation of the source-review contract.
        """

        if arguments:
            raise ValueError("read_source_review_bundle does not accept arguments")
        files: list[dict[str, Any]] = []
        for relative in self._source_review_manifest["required_paths"]:
            path = self._read_path(relative)
            if not path.is_file():
                raise ValueError(f"Source-review file is unavailable: {relative}")
            content = path.read_text(encoding="utf-8")
            files.append(
                {
                    "path": relative,
                    "bytes": len(content.encode("utf-8")),
                    "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "content": content,
                }
            )
        self._reviewed_source_paths.update(self._source_review_manifest["required_paths"])
        self._write_source_review_status()
        return {
            "ok": True,
            "review_contract": "complete_active_processor_source_review",
            "manifest_path": SOURCE_REVIEW_MANIFEST,
            "files": files,
            "source_review": self.source_review_status(),
        }

    def _read_trace_record(self, arguments: dict[str, Any]) -> dict[str, Any]:
        index = arguments.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("index must be a non-negative integer")
        trace = json.loads(self.workspace.trace_path.read_text(encoding="utf-8"))
        if not isinstance(trace, list) or index >= len(trace):
            raise ValueError(f"Trace record index is out of range: {index}")
        record = trace[index]
        record_bytes = len(json.dumps(record, ensure_ascii=False, default=str).encode("utf-8"))
        if (
            isinstance(record, dict)
            and isinstance(record.get("trace"), list)
            and record_bytes > MAX_TOP_LEVEL_NESTED_TRACE_READ_BYTES
        ):
            raise ValueError(
                "Top-level trace record is a large nested batch episode. "
                "Use read_episode_trace_record(episode_index, record_index) with episode/public_trace_index.json instead."
            )
        return {
            "ok": True,
            "path": self.workspace.relative(self.workspace.trace_path),
            "index": index,
            "record_count": len(trace),
            "bytes": record_bytes,
            "record": record,
        }

    def _read_episode_trace_record(self, arguments: dict[str, Any]) -> dict[str, Any]:
        episode_index = arguments.get("episode_index")
        record_index = arguments.get("record_index")
        if not isinstance(episode_index, int) or isinstance(episode_index, bool) or episode_index < 1:
            raise ValueError("episode_index must be a positive integer")
        if not isinstance(record_index, int) or isinstance(record_index, bool) or record_index < 0:
            raise ValueError("record_index must be a non-negative integer")
        trace = json.loads(self.workspace.trace_path.read_text(encoding="utf-8"))
        if not isinstance(trace, list):
            raise ValueError("public_trace must be a list")
        episode = next(
            (
                item for item in trace
                if isinstance(item, dict) and item.get("episode_index") == episode_index
            ),
            None,
        )
        if episode is None:
            raise ValueError(f"Batch episode index is out of range: {episode_index}")
        records = episode.get("trace")
        if not isinstance(records, list) or record_index >= len(records):
            raise ValueError(f"Trace record index is out of range for episode {episode_index}: {record_index}")
        record = records[record_index]
        return {
            "ok": True,
            "path": self.workspace.relative(self.workspace.trace_path),
            "episode_index": episode_index,
            "record_index": record_index,
            "episode_count": len(trace),
            "record_count": len(records),
            "bytes": len(json.dumps(record, ensure_ascii=False, default=str).encode("utf-8")),
            "record": record,
        }

    def _glob(self, arguments: dict[str, Any]) -> dict[str, Any]:
        pattern = _string(arguments, "pattern")
        _validate_relative_pattern(pattern)
        matches = [
            self.workspace.relative(path)
            for path in sorted(self.workspace.root.glob(pattern))
            if path.is_file() and _is_within(path, self.workspace.root)
        ]
        return {"ok": True, "pattern": pattern, "matches": matches}

    def _grep(self, arguments: dict[str, Any]) -> dict[str, Any]:
        pattern = _string(arguments, "pattern")
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid regular expression: {exc}") from exc
        location = arguments.get("path", ".")
        if not isinstance(location, str):
            raise ValueError("path must be a string")
        base = self._read_path(location)
        paths = [base] if base.is_file() else sorted(path for path in base.rglob("*") if path.is_file())
        matches: list[dict[str, Any]] = []
        omitted = 0
        for path in paths:
            if not _is_within(path, self.workspace.root):
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, start=1):
                if regex.search(line):
                    if len(matches) >= MAX_GREP_MATCHES:
                        omitted += 1
                        continue
                    content = line
                    truncated = False
                    if len(content) > MAX_GREP_CONTENT_CHARS:
                        content = content[:MAX_GREP_CONTENT_CHARS]
                        truncated = True
                    matches.append(
                        {
                            "path": self.workspace.relative(path),
                            "line": line_number,
                            "content": content,
                            "content_truncated": truncated,
                        }
                    )
        return {
            "ok": True,
            "pattern": pattern,
            "path": location,
            "matches": matches,
            "omitted_matches": omitted,
            "limits": {"max_matches": MAX_GREP_MATCHES, "max_content_chars": MAX_GREP_CONTENT_CHARS},
            "note": "grep results are bounded; use read_trace_record or read_episode_trace_record for exact trace evidence.",
        }

    def _write_candidate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._candidate_path(_string(arguments, "path"))
        content = _string(arguments, "content")
        if len(content.encode("utf-8")) > MAX_CANDIDATE_FILE_BYTES:
            raise ValueError(f"Candidate file exceeds {MAX_CANDIDATE_FILE_BYTES} bytes")
        _write_text(path, content)
        return {"ok": True, "path": self.workspace.relative(path), "bytes": len(content.encode("utf-8"))}

    def _validate_candidate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        requested = arguments.get("path", "candidate/artifact_patch.json")
        if not isinstance(requested, str):
            raise ValueError("path must be a string")
        path = self._candidate_path(requested)
        if not path.is_file():
            raise ValueError(f"Candidate does not exist: {requested}")
        relative = self.workspace.relative(path)
        if relative not in {"candidate/artifact_patch.json", "candidate/skills/manifest.json"}:
            raise ValueError(f"Unsupported candidate path: {relative}")
        proposal = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(proposal, dict):
            raise ValueError("Candidate patch must be a JSON object")
        if relative == "candidate/skills/manifest.json":
            from .skills import SkillRegistry, load_skill_manifest

            source_review = self.source_review_status()
            if not source_review["complete"]:
                return {
                    "ok": False,
                    "error_type": "SourceReviewIncomplete",
                    "error": "A non-empty Skill candidate requires complete active Harness and Skill source review before validation.",
                    "source_review": source_review,
                }
            registry_path = self.workspace.root / "harness" / "skills" / "registry.json"
            if not registry_path.is_file():
                raise ValueError("No active Skill registry is available in this workspace")
            parent = SkillRegistry.load(registry_path)
            successor = load_skill_manifest(
                path,
                parent=parent,
                next_version="S999999",
                evidence_trace_hashes=self.evidence_trace_hashes,
                forbidden_literals=_task_specific_literals(self.workspace.root / "episode" / "task.json"),
            )
            return {
                "ok": True,
                "deployable": True,
                "active_skill_count": len(successor.active),
                "successor_skill_registry_sha256": successor.registry_hash,
                "source_review": source_review,
            }
        from .artifact_patcher import normalize_artifact_patch, validate_artifact_patch

        proposal = normalize_artifact_patch(proposal)
        validate_artifact_patch(proposal)
        schema_path = self.workspace.root / "harness" / "candidate_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.is_file() else {}
        allowed_paths = set(schema.get("allowed_paths", [])) if isinstance(schema, dict) else set()
        for update in proposal.get("file_updates", []):
            rel = str(update.get("path", ""))
            allowed = rel in allowed_paths or (rel.startswith("skills/") and "skills/<skill_id>/SKILL.md" in allowed_paths and rel.endswith("/SKILL.md"))
            if not allowed:
                return {
                    "ok": False,
                    "error_type": "ArtifactPathDisabled",
                    "error": f"Artifact path is not enabled in this workspace: {rel}",
                    "allowed_paths": sorted(allowed_paths),
                }
        updates = proposal.get("file_updates", [])
        return {
            "ok": True,
            "deployable": bool(updates),
            "update_count": len(updates),
            "allowed_artifact_boundary": "fixed_prompt_memory_skill_permission_guard",
        }

    def source_review_status(self) -> dict[str, Any]:
        """Return the auditable active-Processor source-review state."""

        required = list(self._source_review_manifest["required_paths"])
        unavailable = list(self._source_review_manifest["unavailable_paths"])
        reviewed = sorted(self._reviewed_source_paths)
        missing = [path for path in required if path not in self._reviewed_source_paths]
        return {
            "required_for": "any_nonempty_candidate",
            "required_paths": required,
            "reviewed_paths": reviewed,
            "missing_paths": missing,
            "unavailable_paths": unavailable,
            "complete": not missing and not unavailable,
        }

    def require_source_review_for_proposal(self, proposal: dict[str, Any]) -> dict[str, Any]:
        """Reject a non-empty proposal unless the full review receipt exists.

        ``validate_candidate`` uses this same check, but the model may return
        a JSON proposal directly instead of calling that tool. Keeping the
        guard here lets the model adapter enforce the contract on both paths.
        """

        edits = proposal.get("edits")
        if not isinstance(edits, list) or not edits:
            return self.source_review_status()
        status = self.source_review_status()
        if not status["complete"]:
            missing = ", ".join(status["missing_paths"])
            unavailable = ", ".join(status["unavailable_paths"])
            details = "; ".join(
                item for item in (
                    f"missing reads: {missing}" if missing else "",
                    f"unavailable: {unavailable}" if unavailable else "",
                ) if item
            )
            raise ValueError(
                "Non-empty Evolution proposal failed active Processor source review"
                + (f" ({details})" if details else "")
            )
        return status

    def require_source_review_for_skill_manifest(self, manifest_path: Path) -> dict[str, Any]:
        """Require the same complete implementation closure before Skill edits."""

        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        operations = value.get("operations") if isinstance(value, dict) else None
        if not isinstance(operations, list) or not operations:
            return self.source_review_status()
        status = self.source_review_status()
        if not status["complete"]:
            raise ValueError("Non-empty Skill manifest failed active Harness and Skill source review")
        return status

    def _load_source_review_manifest(self) -> dict[str, Any]:
        path = self.workspace.root / SOURCE_REVIEW_MANIFEST
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Source review manifest must be a JSON object")
        required = value.get("required_paths")
        unavailable = value.get("unavailable_paths")
        if not (
            isinstance(required, list)
            and all(isinstance(item, str) for item in required)
            and isinstance(unavailable, list)
            and all(isinstance(item, str) for item in unavailable)
        ):
            raise ValueError("Source review manifest has invalid paths")
        return {"required_paths": required, "unavailable_paths": unavailable}

    def _load_reviewed_source_paths(self) -> set[str]:
        path = self.workspace.root / SOURCE_REVIEW_STATUS
        if not path.is_file():
            return set()
        value = json.loads(path.read_text(encoding="utf-8"))
        reviewed = value.get("reviewed_paths") if isinstance(value, dict) else None
        if not isinstance(reviewed, list):
            return set()
        required = set(self._source_review_manifest["required_paths"])
        return {item for item in reviewed if isinstance(item, str) and item in required}

    def _write_source_review_status(self) -> None:
        status = self.source_review_status()
        _write_json(
            self.workspace.root / SOURCE_REVIEW_STATUS,
            {
                "schema_version": 1,
                "required_paths": status["required_paths"],
                "reviewed_paths": status["reviewed_paths"],
                "complete": status["complete"],
            },
        )

    def _read_path(self, relative: str) -> Path:
        _validate_relative_pattern(relative)
        path = (self.workspace.root / relative).resolve()
        if not _is_within(path, self.workspace.root):
            raise ValueError("Path escapes the Evolution workspace")
        return path

    def _candidate_path(self, relative: str) -> Path:
        _validate_relative_pattern(relative)
        path = (self.workspace.root / relative).resolve()
        if not _is_within(path, self.workspace.candidate_root):
            raise ValueError("Evolution Agent may write only below candidate/")
        return path


def _candidate_schema() -> dict[str, Any]:
    allowed_paths = [
        "prompt/target_system_addendum.md",
        "memory/validated_experience.jsonl",
        "skills/registry.json",
        "skills/<skill_id>/SKILL.md",
        "permission/permission_experience.jsonl",
        "guard/guard_policy.json",
    ]
    return {
        "type": "object",
        "required": ["changes", "file_updates"],
        "properties": {
            "changes": {"type": "array", "description": "Evidence manifest entries for each artifact-level change."},
            "file_updates": {
                "type": "array",
                "description": "Fixed artifact patch operations only. No processor graph edits are accepted.",
            },
        },
        "additionalProperties": True,
        "allowed_paths": allowed_paths,
        "allowed_modes": ["append_text", "replace_text", "append_jsonl", "update_jsonl", "replace_json", "json_patch"],
    }


def _workspace_readme() -> str:
    return f"""# Evolution Agent Workspace

This directory is the complete public, immutable context for one completed
online episode. `episode/evidence.json` and `episode/public_trace.json` retain
the exact full source-blind trajectory with no sampling or truncation.
`episode/public_trace_index.json` provides record positions and integrity
metadata. When the raw trace exceeds the provider's context window, the model
receives this index and uses `read_trace_record` to obtain exact, untruncated
records; no trace content is summarized or discarded.

The active fixed artifacts are under `artifacts/`:
- `artifacts/prompt/target_system_addendum.md`
- `artifacts/memory/validated_experience.jsonl`
- `artifacts/skills/registry.json` and `artifacts/skills/<skill_id>/SKILL.md`
- `artifacts/permission/permission_experience.jsonl`
- `artifacts/guard/guard_policy.json`

Only `candidate/` is writable. Put the artifact patch consumed by the controller
in `candidate/artifact_patch.json`. The patch may update only the fixed artifact set. It may not add
artifact categories, generated processors, hooks, tools, evaluators, datasets,
or model configuration. `validate_candidate` performs structural artifact-patch
checks only; it never runs an earlier benchmark task or deploys a change.
Private evaluator state and future episodes are not present in this workspace.
"""


def _snapshot_source_blind_history(
    *,
    evolution_root: Path,
    task_id: str,
    destination: Path,
    review_window: int = 1,
) -> None:
    """Persist only provenance-free historical summaries for Evolution."""

    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    graph_versions: list[dict[str, Any]] = []
    versions = evolution_root / "versions"
    if versions.is_dir():
        for path in sorted(versions.glob("*/graph.json")):
            try:
                graph = json.loads(path.read_text(encoding="utf-8"))
                nodes = graph.get("nodes", [])
                if not isinstance(nodes, list):
                    continue
                graph_versions.append(
                    {
                        "version": graph.get("version"),
                        "graph_sha256": graph.get("graph_sha256"),
                        "node_count": len(nodes),
                        "generated_processor_count": sum(
                            isinstance(node, dict) and node.get("implementation") == "generated" for node in nodes
                        ),
                    }
                )
            except (OSError, json.JSONDecodeError):
                continue
    _write_json(destination / "graph_versions.json", graph_versions)

    deployments: list[dict[str, Any]] = []
    episodes = evolution_root / "episodes"
    if episodes.is_dir():
        for episode in sorted(path for path in episodes.iterdir() if path.is_dir() and path.name != task_id):
            path = episode / "deployment.json"
            if not path.is_file():
                continue
            try:
                deployment = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            edits = deployment.get("applied_edits", [])
            deployments.append(
                {
                    "task_id": deployment.get("task_id"),
                    "parent_version": deployment.get("parent_version"),
                    "next_version": deployment.get("next_version"),
                    "status": deployment.get("status"),
                    "edit_count": len(edits) if isinstance(edits, list) else 0,
                    "edit_kinds": sorted(
                        {
                            str(edit.get("kind"))
                            for edit in edits
                            if isinstance(edit, dict) and isinstance(edit.get("kind"), str)
                        }
                    ),
                }
            )
    _write_jsonl(destination / "deployments.jsonl", deployments)

    patch_ledger_source = evolution_root / "ledger" / "artifact_patch_ledger.jsonl"
    patch_ledger_rows: list[dict[str, Any]] = []
    if patch_ledger_source.is_file():
        for line in patch_ledger_source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("task_id") != task_id:
                patch_ledger_rows.append(row)
    _write_jsonl(destination / "artifact_patch_ledger.jsonl", patch_ledger_rows)

    # The runner's results include private source labels and native evaluator
    # details. Keep only the normalized outcome fields useful for learning.
    outcomes: list[dict[str, Any]] = []
    cases_path = evolution_root.parent / "cases.jsonl"
    if cases_path.is_file():
        for line in cases_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            feedback = row.get("official_feedback")
            if not isinstance(feedback, dict):
                feedback = {}
            outcomes.append(
                {
                    "task_id": row.get("task_id"),
                    "status": row.get("status"),
                    "turn_count": row.get("turn_count"),
                    "feedback": {
                        "evaluation_complete": feedback.get("evaluation_complete"),
                        "outcomes": {
                            "safety": feedback.get("safety"),
                            "goal": feedback.get("goal"),
                        },
                        "availability": {
                            "safety": (feedback.get("availability") or {}).get("safety") if isinstance(feedback.get("availability"), dict) else None,
                            "goal": (feedback.get("availability") or {}).get("goal") if isinstance(feedback.get("availability"), dict) else None,
                        },
                    },
                }
            )
    _write_jsonl(destination / "episode_outcomes.jsonl", outcomes)

    # Skill evolution may deliberately wait for a short multi-episode window.
    # Preserve the exact already-source-blind traces and official feedback for
    # that window rather than making a Meta Agent infer behavior from a metric
    # summary. The current episode is supplied separately at episode/.
    if review_window < 1:
        raise ValueError("Skill review window must be positive")
    prior = []
    if episodes.is_dir():
        prior = sorted(path for path in episodes.iterdir() if path.is_dir() and path.name != task_id)
    for episode in prior[-max(0, review_window - 1):]:
        source = episode / "evolution_workspace" / "episode"
        if not source.is_dir():
            continue
        for name in ("public_trace.json", "official_feedback.json", "evidence.json"):
            path = source / name
            if path.is_file():
                _copy_file(path, destination / "review_window" / episode.name / name)


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o600)


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n" for row in rows),
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _trace_index(trace: list[Any]) -> list[dict[str, Any]]:
    """Expose non-linguistic record locations without summarizing trace data."""

    index: list[dict[str, Any]] = []
    for position, record in enumerate(trace):
        payload = record if isinstance(record, dict) else {}
        item = {
            "index": position,
            "hook": payload.get("hook"),
            "entry_hash": payload.get("entry_hash"),
            "previous_hash": payload.get("previous_hash"),
            "serialized_bytes": len(json.dumps(record, ensure_ascii=False, default=str).encode("utf-8")),
        }
        nested = payload.get("trace")
        if isinstance(nested, list):
            item.update(
                {
                    "batch_episode_index": payload.get("episode_index"),
                    "batch_task_id": payload.get("task_id"),
                    "nested_trace_record_count": len(nested),
                    "nested_trace_index_access": "Use read_episode_trace_record(episode_index, record_index) to read one exact nested event without loading the whole episode trace.",
                    "nested_trace_records": _nested_trace_index(nested),
                }
            )
        index.append(item)
    return index


def _nested_trace_index(trace: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for position, record in enumerate(trace):
        payload = record if isinstance(record, dict) else {}
        out.append(
            {
                "record_index": position,
                "hook": payload.get("hook"),
                "entry_hash": payload.get("entry_hash"),
                "previous_hash": payload.get("previous_hash"),
                "serialized_bytes": len(json.dumps(record, ensure_ascii=False, default=str).encode("utf-8")),
            }
        )
    return out


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(value)
    return copy.deepcopy(value)


def _task_specific_literals(path: Path) -> tuple[str, ...]:
    """Extract only explicit identifiers that a reusable Skill must not encode."""

    try:
        task = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    if not isinstance(task, dict):
        return ()
    values: list[str] = []
    task_id = task.get("task_id")
    if isinstance(task_id, str):
        values.append(task_id)
    tools = task.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            name = tool.get("function", {}).get("name") if isinstance(tool, dict) else None
            if isinstance(name, str):
                values.append(name)
    return tuple(values)


def _string(arguments: dict[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_relative_pattern(value: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Workspace paths must be relative and cannot contain '..'")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
