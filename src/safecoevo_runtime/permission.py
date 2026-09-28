"""High-level permission experience used by the Safety Harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .contracts import PermissionExperienceItem


class PermissionExperienceStore(Protocol):
    def retrieve(self, query: str, *, limit: int) -> list[PermissionExperienceItem]: ...

    def append(self, item: PermissionExperienceItem) -> None: ...


@dataclass
class InMemoryPermissionExperienceStore:
    """Verified permission lessons with deterministic recent-first retrieval."""

    items: list[PermissionExperienceItem] = field(default_factory=list)

    def retrieve(self, query: str, *, limit: int) -> list[PermissionExperienceItem]:
        if limit < 1:
            raise ValueError("Permission experience retrieval limit must be positive")
        del query  # Permission lessons are deliberately broad, not task-specific rules.
        return [item for item in reversed(self.items) if item.verified][:limit]

    def append(self, item: PermissionExperienceItem) -> None:
        if not item.verified:
            raise ValueError("Only verified permission experience may enter the permission store")
        if any(existing.permission_id == item.permission_id for existing in self.items):
            raise ValueError(f"Duplicate permission_id: {item.permission_id}")
        self.items.append(item)
