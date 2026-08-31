"""Storage abstraction for context items."""

from datetime import datetime
from typing import Protocol

from contextos.models import ContextEdge, ContextItem, ContextType, LifecycleTier


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
        updated_after: datetime | None = None,
        updated_before: datetime | None = None,
    ) -> list[ContextItem]:
        """Return items matching optional filters in deterministic ID order."""
        ...

    def save_edge(self, edge: ContextEdge) -> None:
        """Create or replace an edge after validating both endpoint IDs."""
        ...

    def load_dependencies(self, item_id: str | None = None) -> list[ContextEdge]:
        """Return all edges, optionally restricted to an incident item."""
        ...

    def update_tier(self, item_id: str, lifecycle_tier: LifecycleTier) -> None:
        """Update one persisted lifecycle tier."""
        ...

    def delete_item(self, item_id: str) -> None:
        """Explicitly delete one item or raise a typed not-found error."""
        ...
