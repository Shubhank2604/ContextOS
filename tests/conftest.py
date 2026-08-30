"""Shared test fixtures."""

from datetime import UTC, datetime

import pytest

from contextos.models import ContextItem, ContextType


@pytest.fixture
def context_item() -> ContextItem:
    """Return a valid optional context item."""
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return ContextItem(
        id="item-1",
        content="A useful context fact.",
        type=ContextType.MEMORY,
        created_at=timestamp,
        updated_at=timestamp,
        importance=0.75,
    )
