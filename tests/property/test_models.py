"""Property tests for normalized model fields."""

from datetime import UTC, datetime

from hypothesis import given
from hypothesis import strategies as st

from contextos.models import ContextItem, ContextType


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_normalized_importance_is_preserved(importance: float) -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    item = ContextItem(
        id="property-item",
        content="Required content",
        type=ContextType.TASK_STATE,
        created_at=timestamp,
        updated_at=timestamp,
        importance=importance,
    )
    assert 0.0 <= item.importance <= 1.0
