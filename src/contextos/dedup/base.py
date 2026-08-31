"""Shared deduplication result models."""

from __future__ import annotations

from dataclasses import dataclass, field

from contextos.models import ContextItem


@dataclass(frozen=True)
class DuplicateMatch:
    """Explain why one optional item was removed as a duplicate."""

    item_id: str
    duplicate_of: str
    reason: str
    similarity: float | None = None


@dataclass
class DeduplicationResult:
    """Surviving items plus deterministic duplicate evidence."""

    items: list[ContextItem]
    matches: list[DuplicateMatch] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def matches_by_item_id(self) -> dict[str, DuplicateMatch]:
        """Index duplicate decisions by removed item ID."""
        return {match.item_id: match for match in self.matches}
