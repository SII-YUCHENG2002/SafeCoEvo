"""Verified-experience Memory interfaces used by the Safety Harness."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .contracts import MemoryItem


# These occur in nearly every benchmark request and feedback lesson. Keeping
# them out of retrieval prevents a generic "task/user/request" overlap from
# injecting an unrelated prior episode into the next agent context.
_RETRIEVAL_STOPWORDS = frozenset(
    {
        "about", "after", "agent", "available", "before", "check", "checking",
        "complete", "completed", "continue", "feedback", "found", "from",
        "goal", "harness", "original", "prior", "provided", "request",
        "safety", "task", "that", "the", "this", "tool", "tools", "user",
        "verified", "while", "with",
    }
)


def memory_identity(item: MemoryItem) -> dict[str, str]:
    """Stable content fingerprint for injected and live memory versions."""

    payload = json.dumps([item.content, sorted(item.tags)], ensure_ascii=False)
    return {"memory_id": item.memory_id, "fingerprint": hashlib.sha256(payload.encode()).hexdigest()}


def _retrieval_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if len(term) >= 4 and term not in _RETRIEVAL_STOPWORDS
    }


_MEMORY_DIRECTION_TAGS = ("dir_release", "dir_restrict", "dir_neutral")
# At most this many release-direction lessons may occupy one retrieval window.
# Direction tags are produced by the Evolution Agent itself at write time
# (mandatory per its system prompt); this is a scheduler-side balance rule over
# system-produced metadata, not operator content classification.
_RELEASE_DIRECTION_PER_WINDOW = 1


def memory_direction(item: MemoryItem) -> str:
    for tag in _MEMORY_DIRECTION_TAGS:
        if tag in (item.tags or ()):
            return tag
    return "dir_neutral"


def _apply_release_direction_cap(matched: list[MemoryItem], limit: int) -> list[MemoryItem]:
    """Keep lexical order but allow at most one release-direction lesson per window."""
    selected: list[MemoryItem] = []
    release_used = 0
    for item in matched:
        if len(selected) >= limit:
            break
        if memory_direction(item) == "dir_release":
            if release_used >= _RELEASE_DIRECTION_PER_WINDOW:
                continue
            release_used += 1
        selected.append(item)
    return selected


class MemoryStore(Protocol):
    def retrieve(self, query: str, *, limit: int) -> list[MemoryItem]: ...

    def append(self, item: MemoryItem) -> None: ...


@dataclass
class InMemoryMemoryStore:
    """Verified memory with deterministic lexical retrieval."""

    items: list[MemoryItem] = field(default_factory=list)
    retrieval_mode: str = "lexical"
    _last_retrieval: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def retrieval_metadata(self) -> dict[str, Any]:
        """Return prompt-free diagnostics for one retrieval attempt."""

        return dict(self._last_retrieval)

    def retrieve(self, query: str, *, limit: int) -> list[MemoryItem]:
        if limit < 1:
            raise ValueError("Memory retrieval limit must be positive")
        return self._retrieve_lexical(query, limit=limit)

    def _retrieve_lexical(self, query: str, *, limit: int) -> list[MemoryItem]:
        terms = _retrieval_terms(query)
        ranked: list[tuple[int, MemoryItem]] = []
        for item in self.items:
            if not item.verified:
                continue
            text = (item.content + " " + " ".join(item.tags)).lower()
            ranked.append((sum(term in text for term in terms), item))
        ordered = sorted(ranked, key=lambda row: (-row[0], row[1].memory_id))
        matched = [item for score, item in ordered if score > 0]
        if matched:
            selected = _apply_release_direction_cap(matched, limit)
            score_by_id = {item.memory_id: score for score, item in ordered}
            self._last_retrieval = {
                "mode": "lexical",
                "candidate_count": len(ranked),
                "returned_count": len(selected),
                "matches": [
                    {"memory_id": item.memory_id, "lexical_score": score_by_id[item.memory_id]}
                    for item in selected
                ],
            }
            return selected
        self._last_retrieval = {
            "mode": "lexical",
            "candidate_count": len(ranked),
            "returned_count": 0,
            "matches": [],
        }
        return []

    def append(self, item: MemoryItem) -> None:
        if not item.verified:
            raise ValueError("Only verified memory may enter the long-term store")
        if any(existing.memory_id == item.memory_id for existing in self.items):
            raise ValueError(f"Duplicate memory_id: {item.memory_id}")
        self.items.append(item)
