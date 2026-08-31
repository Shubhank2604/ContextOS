"""Property tests for normalized composite scoring."""

from datetime import UTC, datetime

from hypothesis import given
from hypothesis import strategies as st

from contextos.config import OptimizationPolicy
from contextos.models import ContextItem, ContextType
from contextos.scoring import composite_scores

_NORMALIZED = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


@given(
    relevance=_NORMALIZED,
    importance=_NORMALIZED,
    recency=_NORMALIZED,
    novelty=_NORMALIZED,
    dependency=_NORMALIZED,
    type_priority=_NORMALIZED,
)
def test_composite_score_always_remains_normalized(
    relevance: float,
    importance: float,
    recency: float,
    novelty: float,
    dependency: float,
    type_priority: float,
) -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    item = ContextItem(
        id="item",
        content="content",
        type=ContextType.MEMORY,
        created_at=timestamp,
        updated_at=timestamp,
        importance=importance,
    )
    result = composite_scores(
        [item],
        policy=OptimizationPolicy(max_input_tokens=100),
        relevance={item.id: relevance},
        importance={item.id: importance},
        recency={item.id: recency},
        novelty={item.id: novelty},
        dependency={item.id: dependency},
        type_priority={item.id: type_priority},
    )[item.id]
    assert 0.0 <= result.composite_utility <= 1.0
