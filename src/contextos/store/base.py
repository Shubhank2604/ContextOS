"""Storage abstraction for context items."""

from typing import Protocol

from contextos.models import ContextItem, ContextType, LifecycleTier


class ContextStore(Protocol):
    """Minimum context-item repository contract."""

    def save_item(self, item: ContextItem) -> None:
        """Create or replace an item by ID."""
        ...

    def load_item(self, item_id: str) -> ContextItem:
        """Load one item or raise a typed not-found error."""
        ...

    def list_items(
        self,
        *,
        lifecycle_tier: LifecycleTier | None = None,
        context_type: ContextType | None = None,
    ) -> list[ContextItem]:
        """Return items matching optional filters in deterministic ID order."""
        ...

    def delete_item(self, item_id: str) -> None:
        """Explicitly delete one item or raise a typed not-found error."""
        ...
