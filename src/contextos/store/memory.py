"""In-memory context store used by local workflows and tests."""

from __future__ import annotations

from datetime import datetime

from contextos.errors import ContextItemNotFoundError, UnknownDependencyReference
from contextos.models import ContextEdge, ContextItem, ContextType, LifecycleTier


class InMemoryContextStore:
    """Deterministic in-process implementation of the context store contract."""

    def __init__(self) -> None:
        self._items: dict[str, ContextItem] = {}
        self._edges: dict[tuple[str, str, str], ContextEdge] = {}

    def save_item(self, item: ContextItem) -> None:
        """Create or replace an item, isolating store state from caller mutation."""
        self._items[item.id] = item.model_copy(deep=True)

    def load_item(self, item_id: str) -> ContextItem:
        """Load an isolated copy of an item."""
        try:
            return self._items[item_id].model_copy(deep=True)
        except KeyError as exc:
            raise ContextItemNotFoundError(f"Context item {item_id!r} was not found") from exc

    def list_items(
        self,
        *,
        lifecycle_tier: LifecycleTier | None = None,
        context_type: ContextType | None = None,
        updated_after: datetime | None = None,
        updated_before: datetime | None = None,
    ) -> list[ContextItem]:
        """Return filtered copies in deterministic ID order."""
        items = (
            item
            for item in self._items.values()
            if (lifecycle_tier is None or item.lifecycle_tier is lifecycle_tier)
            and (context_type is None or item.type is context_type)
            and (updated_after is None or item.updated_at >= updated_after)
            and (updated_before is None or item.updated_at <= updated_before)
        )
        return [item.model_copy(deep=True) for item in sorted(items, key=lambda value: value.id)]

    def delete_item(self, item_id: str) -> None:
        """Delete exactly one requested item."""
        if item_id not in self._items:
            raise ContextItemNotFoundError(f"Context item {item_id!r} was not found")
        del self._items[item_id]
        self._edges = {
            key: edge
            for key, edge in self._edges.items()
            if edge.source_id != item_id and edge.target_id != item_id
        }

    def save_edge(self, edge: ContextEdge) -> None:
        """Save an edge only when both endpoints exist."""
        unknown = sorted({edge.source_id, edge.target_id} - self._items.keys())
        if unknown:
            raise UnknownDependencyReference(
                f"Unknown dependency endpoint IDs: {', '.join(unknown)}"
            )
        key = (edge.source_id, edge.target_id, edge.relation.value)
        self._edges[key] = edge.model_copy(deep=True)

    def load_dependencies(self, item_id: str | None = None) -> list[ContextEdge]:
        """Load isolated edges in deterministic endpoint order."""
        edges = (
            edge
            for edge in self._edges.values()
            if item_id is None or edge.source_id == item_id or edge.target_id == item_id
        )
        return [
            edge.model_copy(deep=True)
            for edge in sorted(
                edges,
                key=lambda value: (
                    value.source_id,
                    value.target_id,
                    value.relation.value,
                ),
            )
        ]

    def update_tier(self, item_id: str, lifecycle_tier: LifecycleTier) -> None:
        """Update a tier while preserving all other stored fields."""
        item = self.load_item(item_id)
        item.lifecycle_tier = lifecycle_tier
        self.save_item(item)
