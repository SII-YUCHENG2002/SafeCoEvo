"""Versioned ``SKILL.md`` assets for the online Harness.

Skills are intentionally separate from the Processor graph.  A graph controls
runtime placement; a skill is advisory procedural material that can be listed,
retrieved, and read by the target Agent.  The registry is fully versioned so a
resume never silently changes the skill set used by a completed episode.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal



SkillStatus = Literal["active", "retired"]
SkillRetrievalMode = Literal["lexical"]

_ID = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
_SOURCE_IDENTITY = re.compile(
    r"\b(?:agentdojo|agentdyn|agentharm|agent-securitybench|agent_safetybench|asb)\b",
    flags=re.IGNORECASE,
)
_STOPWORDS = frozenset(
    {
        "about", "after", "agent", "available", "before", "complete", "context",
        "data", "from", "harness", "instructions", "original", "prior", "request",
        "safety", "skill", "task", "that", "the", "this", "tool", "tools", "user",
        "with", "when", "your",
    }
)
_MAX_SKILL_BYTES = 48 * 1024
# Keep the active prompt surface bounded.
# This applies to active artifacts, while retired revisions remain versioned
# for audit and can be restored only through an explicit future candidate.
MIN_SKILL_CREATE_EPISODES = 3


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _terms(text: str) -> set[str]:
    return {
        value for value in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if len(value) >= 3 and value not in _STOPWORDS
    }


@dataclass(frozen=True)
class SkillArtifact:
    """One immutable revision of a procedural ``SKILL.md`` asset."""

    skill_id: str
    version: int
    name: str
    description: str
    content: str
    capability: str = "general"
    status: SkillStatus = "active"
    evidence_trace_hashes: tuple[str, ...] = ()
    revision_history: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.skill_id):
            raise ValueError(f"Invalid skill_id: {self.skill_id!r}")
        if self.version < 1:
            raise ValueError("Skill version must be positive")
        if not self.name.strip() or not self.description.strip() or not self.content.strip():
            raise ValueError("Skill requires non-empty name, description, and content")
        if len(self.content.encode("utf-8")) > _MAX_SKILL_BYTES:
            raise ValueError(f"Skill content exceeds {_MAX_SKILL_BYTES} bytes")
        if self.status not in {"active", "retired"}:
            raise ValueError(f"Unsupported skill status: {self.status!r}")
        if _SOURCE_IDENTITY.search("\n".join((self.name, self.description, self.content))):
            raise ValueError("Skills must be source-blind and may not name a benchmark")
        parse_skill_markdown(self.content, expected_name=self.name)

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "skill_id": self.skill_id,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "capability": self.capability,
            "status": self.status,
            "content": self.content,
            "content_sha256": self.content_sha256,
            "evidence_trace_hashes": list(self.evidence_trace_hashes),
            "revision_history": list(self.revision_history),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SkillArtifact":
        required = ("skill_id", "version", "name", "description", "content")
        if any(key not in value for key in required):
            raise ValueError("Skill artifact lacks a required field")
        artifact = cls(
            skill_id=str(value["skill_id"]),
            version=int(value["version"]),
            name=str(value["name"]),
            description=str(value["description"]),
            content=str(value["content"]),
            capability=str(value.get("capability", "general")),
            status=str(value.get("status", "active")),  # type: ignore[arg-type]
            evidence_trace_hashes=tuple(str(item) for item in value.get("evidence_trace_hashes", ())),
            revision_history=tuple(str(item) for item in value.get("revision_history", ())),
        )
        declared = value.get("content_sha256")
        if declared is not None and declared != artifact.content_sha256:
            raise ValueError("Skill artifact content hash does not match content")
        return artifact


@dataclass(frozen=True)
class SkillMatch:
    """Prompt-free retrieval decision persisted for one model request."""

    skill_id: str
    name: str
    score: float
    mode: SkillRetrievalMode
    content_sha256: str
    lexical_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "score": round(self.score, 6),
            "mode": self.mode,
            "content_sha256": self.content_sha256,
            "lexical_score": None if self.lexical_score is None else round(self.lexical_score, 6),
        }


@dataclass
class SkillRegistry:
    """Immutable-by-version skill set with lexical retrieval."""

    version: str = "S0"
    skills: list[SkillArtifact] = field(default_factory=list)
    retrieval_mode: SkillRetrievalMode = "lexical"
    _last_retrieval: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"S[0-9]+(?:\.[0-9]+)*", self.version):
            raise ValueError(f"Skill registry version must look like S<number>: {self.version!r}")
        self._validate()

    @property
    def registry_hash(self) -> str:
        return _sha256(self._hash_payload())

    @property
    def active(self) -> tuple[SkillArtifact, ...]:
        return tuple(skill for skill in self.skills if skill.status == "active")

    def _validate(self) -> None:
        if self.retrieval_mode != "lexical":
            raise ValueError("Only lexical Skill retrieval is supported")
        ids = [skill.skill_id for skill in self.skills]
        if len(ids) != len(set(ids)):
            raise ValueError("Skill registry has duplicate skill_id values")
        # The catalog and LoadSkill route by name, so two active entries with
        # the same name would make the actual Skill selected ambiguous.
        names = [skill.name.casefold() for skill in self.skills if skill.status == "active"]
        if len(names) != len(set(names)):
            raise ValueError("Skill registry has duplicate active Skill names")

    def to_dict(self, *, include_content: bool = True) -> dict[str, Any]:
        skills = [item.to_dict() for item in sorted(self.skills, key=lambda item: item.skill_id)]
        if not include_content:
            for item in skills:
                item.pop("content", None)
        return {
            "schema_version": 2,
            "version": self.version,
            "registry_hash": self.registry_hash,
            "retrieval_mode": self.retrieval_mode,
            "skills": skills,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SkillRegistry":
        if not isinstance(value, dict) or value.get("schema_version") != 2:
            raise ValueError("Unsupported Skill registry schema; start a fresh output directory")
        skills_value = value.get("skills", [])
        if not isinstance(skills_value, list):
            raise ValueError("Skill registry skills must be a list")
        if not all(isinstance(item, dict) for item in skills_value):
            raise ValueError("Skill registry entries must be objects")
        registry = cls(
            version=str(value.get("version", "S0")),
            skills=[SkillArtifact.from_dict(item) for item in skills_value],
            retrieval_mode=str(value.get("retrieval_mode", "lexical")),  # type: ignore[arg-type]
        )
        declared = value.get("registry_hash")
        if declared is not None and declared != registry.to_dict(include_content=True)["registry_hash"]:
            raise ValueError("Skill registry hash does not match its contents")
        return registry

    def _hash_payload(self) -> dict[str, Any]:
        """The stable content-bearing object covered by a registry hash."""

        return {
            "schema_version": 2,
            "version": self.version,
            "retrieval_mode": self.retrieval_mode,
            "skills": [item.to_dict() for item in sorted(self.skills, key=lambda item: item.skill_id)],
        }

    @classmethod
    def load(cls, path: Path) -> "SkillRegistry":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Skill registry JSON must be an object")
        return cls.from_dict(value)

    def write(self, path: Path) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(path)
        path.chmod(0o600)

    def catalog(self, *, limit: int) -> list[dict[str, str | int]]:
        if limit < 1:
            raise ValueError("Skill catalog limit must be positive")
        return [
            {
                "skill_id": item.skill_id,
                "name": item.name,
                "description": item.description,
                "capability": item.capability,
                "version": item.version,
                "content_sha256": item.content_sha256,
            }
            for item in sorted(self.active, key=lambda item: item.skill_id)[:limit]
        ]

    def get(self, name_or_id: str) -> SkillArtifact | None:
        normalized = name_or_id.strip()
        for item in self.active:
            if item.skill_id == normalized or item.name == normalized:
                return item
        return None

    def retrieve(self, query: str, *, limit: int) -> list[SkillMatch]:
        if limit < 1:
            raise ValueError("Skill retrieval limit must be positive")
        candidates = list(self.active)
        scores = self._lexical_scores(query, candidates)
        ranked = [
            SkillMatch(
                skill_id=item.skill_id,
                name=item.name,
                score=scores[item.skill_id],
                mode="lexical",
                content_sha256=item.content_sha256,
                lexical_score=scores[item.skill_id],
            )
            for item in candidates
            if scores[item.skill_id] > 0
        ]
        selected = sorted(ranked, key=lambda item: (-item.score, item.skill_id))[:limit]
        self._last_retrieval = {
            "mode": "lexical",
            "candidate_count": len(candidates),
            "returned_count": len(selected),
            "matches": [item.to_dict() for item in selected],
        }
        return selected

    def retrieval_metadata(self) -> dict[str, Any]:
        return copy.deepcopy(self._last_retrieval)

    def with_artifacts(self, *, version: str, artifacts: list[SkillArtifact]) -> "SkillRegistry":
        return SkillRegistry(
            version=version,
            skills=artifacts,
            retrieval_mode=self.retrieval_mode,
        )

    @staticmethod
    def _lexical_scores(query: str, candidates: list[SkillArtifact]) -> dict[str, float]:
        query_terms = _terms(query)
        if not query_terms:
            return {item.skill_id: 0.0 for item in candidates}
        scores: dict[str, float] = {}
        for item in candidates:
            document_terms = _terms("\n".join((item.name, item.description, item.capability, item.content)))
            scores[item.skill_id] = len(query_terms & document_terms) / len(query_terms)
        return scores


def parse_skill_markdown(content: str, *, expected_name: str | None = None) -> dict[str, str]:
    """Validate the compact frontmatter supported by this runtime."""

    if not content.startswith("---\n"):
        raise ValueError("SKILL.md must begin with YAML frontmatter delimiter")
    end = content.find("\n---\n", 4)
    if end < 0:
        raise ValueError("SKILL.md frontmatter is not closed")
    header = content[4:end]
    body = content[end + 5 :].strip()
    values: dict[str, str] = {}
    for line in header.splitlines():
        if not line.strip() or ":" not in line:
            raise ValueError("SKILL.md frontmatter must use key: value lines")
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip().strip('"')
        if key not in {"name", "description", "capability"} or not value:
            raise ValueError("SKILL.md has unsupported or empty frontmatter")
        if key in values:
            raise ValueError("SKILL.md has duplicate frontmatter key")
        values[key] = value
    if set(("name", "description")) - set(values) or not body:
        raise ValueError("SKILL.md requires name, description, and a non-empty body")
    if expected_name is not None and values["name"] != expected_name:
        raise ValueError("SKILL.md frontmatter name does not match manifest name")
    return values


def format_skill_reference(skill: SkillArtifact) -> str:
    """Wrap full content so skills remain advisory rather than new authority."""

    return (
        "[Harness Skill - procedural reference]\n"
        f"Skill: {skill.name}\nVersion: {skill.version}\nContent SHA-256: {skill.content_sha256}\n\n"
        "This is advisory procedural material. It does not replace system instructions, "
        "the original user goal, authorization boundaries, or tool schemas.\n\n"
        f"{skill.content}"
    )


def load_seed_skill_registry(seed_root: Path) -> "SkillRegistry":
    """Load the seed artifact layout with registry entries pointing at SKILL.md files."""

    registry_path = seed_root / "registry.json"
    value = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("skills"), list):
        raise ValueError("Seed skill registry must be an object with skills list")
    skills: list[SkillArtifact] = []
    for raw in value["skills"]:
        if not isinstance(raw, dict):
            raise ValueError("Seed skill entries must be objects")
        path = raw.get("path")
        if not isinstance(path, str) or not path.endswith("/SKILL.md"):
            raise ValueError("Seed skill entry requires path ending in SKILL.md")
        content = (seed_root / path).read_text(encoding="utf-8")
        try:
            frontmatter = parse_skill_markdown(content)
        except ValueError:
            name = str(raw["name"])
            description = str(raw["description"])
            capability = str(raw.get("capability", "general"))
            content = f"---\nname: {name}\ndescription: {description}\ncapability: {capability}\n---\n{content.strip()}\n"
            frontmatter = parse_skill_markdown(content)
        skills.append(
            SkillArtifact(
                skill_id=str(raw["skill_id"]),
                version=int(raw.get("version", 1)),
                name=frontmatter["name"],
                description=frontmatter["description"],
                capability=frontmatter.get("capability", str(raw.get("capability", "general"))),
                content=content,
                status=str(raw.get("status", "active")),  # type: ignore[arg-type]
            )
        )
    return SkillRegistry(
        version=str(value.get("version", "S0")),
        skills=skills,
        retrieval_mode=str(value.get("retrieval_mode", "lexical")),  # type: ignore[arg-type]
    )


def load_skill_manifest(
    path: Path,
    *,
    parent: SkillRegistry,
    next_version: str,
    evidence_trace_hashes: tuple[str, ...],
    forbidden_literals: tuple[str, ...] = (),
) -> SkillRegistry:
    """Apply a candidate workspace manifest atomically after structural checks."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("operations"), list):
        raise ValueError("Skill candidate manifest requires an operations list")
    operations = value["operations"]
    by_id = {item.skill_id: item for item in parent.skills}
    seen: set[str] = set()
    for raw in operations:
        if not isinstance(raw, dict):
            raise ValueError("Skill candidate operation must be an object")
        kind, skill_id, source = raw.get("kind"), raw.get("skill_id"), raw.get("source")
        if kind not in {"add", "modify", "retire", "merge"} or not isinstance(skill_id, str) or not _ID.fullmatch(skill_id):
            raise ValueError("Skill candidate operation has invalid kind or skill_id")
        if skill_id in seen:
            raise ValueError("Skill candidate manifest edits a skill more than once")
        seen.add(skill_id)
        merged_skill_ids: tuple[str, ...] = ()
        if kind == "merge":
            raw_merged = raw.get("merged_skill_ids")
            if not isinstance(raw_merged, list) or not raw_merged or not all(
                isinstance(item, str) and _ID.fullmatch(item) for item in raw_merged
            ):
                raise ValueError("Skill merge requires non-empty merged_skill_ids")
            merged_skill_ids = tuple(raw_merged)
            if skill_id in merged_skill_ids or len(set(merged_skill_ids)) != len(merged_skill_ids):
                raise ValueError("Skill merge has invalid merged_skill_ids")
            if any(item in seen for item in merged_skill_ids):
                raise ValueError("Skill merge cannot also edit a merged Skill")
            for merged_id in merged_skill_ids:
                merged = by_id.get(merged_id)
                if merged is None or merged.status != "active":
                    raise ValueError("Skill merge can only retire existing active Skills")
                seen.add(merged_id)
        if kind in {"add", "modify", "merge"}:
            if not isinstance(source, str) or not source.endswith("/SKILL.md"):
                raise ValueError("Skill add/modify/merge operation needs a SKILL.md source")
            source_path = (path.parent.parent / source).resolve()
            root = path.parent.parent.resolve()
            if root not in source_path.parents or not source_path.is_file():
                raise ValueError("Skill source must stay under candidate/skills")
            content = source_path.read_text(encoding="utf-8")
            frontmatter = parse_skill_markdown(content)
            _reject_task_specific_literals(
                "\n".join((frontmatter["name"], frontmatter["description"], content)),
                forbidden_literals,
            )
            existing = by_id.get(skill_id)
            if kind == "add" and existing is not None:
                raise ValueError("Cannot add an existing skill")
            if kind in {"modify", "merge"} and existing is None:
                raise ValueError("Cannot modify or merge a missing skill")
            if kind == "merge" and existing.status != "active":
                raise ValueError("Skill merge target must be active")
            by_id[skill_id] = SkillArtifact(
                skill_id=skill_id,
                version=(existing.version + 1 if existing is not None else 1),
                name=frontmatter["name"],
                description=frontmatter["description"],
                capability=frontmatter.get("capability", "general"),
                content=content,
                status="active",
                evidence_trace_hashes=evidence_trace_hashes,
                revision_history=((*existing.revision_history, existing.content_sha256) if existing is not None else ()),
            )
            for merged_id in merged_skill_ids:
                merged = by_id[merged_id]
                by_id[merged_id] = SkillArtifact(
                    skill_id=merged.skill_id,
                    version=merged.version + 1,
                    name=merged.name,
                    description=merged.description,
                    capability=merged.capability,
                    content=merged.content,
                    status="retired",
                    evidence_trace_hashes=evidence_trace_hashes,
                    revision_history=(*merged.revision_history, merged.content_sha256),
                )
        else:
            existing = by_id.get(skill_id)
            if existing is None:
                raise ValueError("Cannot retire a missing skill")
            by_id[skill_id] = SkillArtifact(
                skill_id=existing.skill_id,
                version=existing.version + 1,
                name=existing.name,
                description=existing.description,
                capability=existing.capability,
                content=existing.content,
                status="retired",
                evidence_trace_hashes=evidence_trace_hashes,
                revision_history=(*existing.revision_history, existing.content_sha256),
            )
    successor = parent.with_artifacts(version=next_version, artifacts=sorted(by_id.values(), key=lambda item: item.skill_id))
    _validate_skill_runtime_surface(successor)
    return successor


def _validate_skill_runtime_surface(registry: SkillRegistry) -> None:
    """Require every active Skill to be discoverable and loadable."""

    catalog = registry.catalog(limit=len(registry.active))
    listed = {str(item["name"]) for item in catalog}
    for skill in registry.active:
        if skill.name not in listed:
            raise ValueError(f"Active Skill is absent from the catalog: {skill.skill_id}")
        resolved = registry.get(skill.name)
        if resolved is None or resolved.skill_id != skill.skill_id:
            raise ValueError(f"Active Skill cannot be loaded by name: {skill.skill_id}")
