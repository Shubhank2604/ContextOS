"""In-memory context store used by local workflows and tests."""

from __future__ import annotations

from contextos.errors import ContextItemNotFoundError
from contextos.models import ContextItem, ContextType, LifecycleTier


class InMemoryContextStore:
    """Deterministic in-process implementation of the context store contract."""

    def __init__(self) -> None:
        self._items: dict[str, ContextItem] = {}

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
    ) -> list[ContextItem]:
        """Return filtered copies in deterministic ID order."""
        items = (
            item
            for item in self._items.values()
            if (lifecycle_tier is None or item.lifecycle_tier is lifecycle_tier)
            and (context_type is None or item.type is context_type)
        )
        return [item.model_copy(deep=True) for item in sorted(items, key=lambda value: value.id)]

    def delete_item(self, item_id: str) -> None:
        """Delete exactly one requested item."""
        if item_id not in self._items:
            raise ContextItemNotFoundError(f"Context item {item_id!r} was not found")
        del self._items[item_id]
