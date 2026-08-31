"""Property tests for allocation budget and mandatory-retention invariants."""

from datetime import UTC, datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st

from contextos.budget import TokenBudgetAllocator, validate_contextual_budget
from contextos.config import OptimizationPolicy
from contextos.errors import MandatoryContextOverflow
from contextos.models import ContextItem, ContextType
from contextos.scoring import ScoreBreakdown


def _item(item_id: str, tokens: int, *, mandatory: bool = False) -> ContextItem:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return ContextItem(
        id=item_id,
        content=item_id,
        type=ContextType.MEMORY,
        created_at=timestamp,
        updated_at=timestamp,
        importance=0.5,
        mandatory=mandatory,
        evictable=not mandatory,
        token_count=tokens,
    )


def _score(value: float) -> ScoreBreakdown:
    return ScoreBreakdown(
        relevance=value,
        importance=value,
        recency=value,
        novelty=value,
        dependency=value,
        type_priority=value,
        composite_utility=value,
    )


@given(
    token_counts=st.lists(st.integers(min_value=1, max_value=50), min_size=1, max_size=8),
    utilities=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        min_size=8,
        max_size=8,
    ),
    budget=st.integers(min_value=1, max_value=150),
)
def test_allocation_plan_never_exceeds_optional_budget(
    token_counts: list[int],
    utilities: list[float],
    budget: int,
) -> None:
    items = [_item(f"item-{index}", tokens) for index, tokens in enumerate(token_counts)]
    scores = {item.id: _score(utilities[index]) for index, item in enumerate(items)}
    plan = TokenBudgetAllocator().allocate(
        items,
        scores=scores,
        policy=OptimizationPolicy(
            max_input_tokens=budget,
            compression_target_ratio=0.5,
            minimum_compressed_tokens=1,
        ),
    )
    assert plan.direct_tokens + plan.reserved_compression_tokens <= plan.optional_budget
    assert len(plan.candidate_item_ids) == (
        len(plan.direct_selected) + len(plan.compression_requests) + len(plan.rejected_item_ids)
    )


@given(
    mandatory_tokens=st.lists(st.integers(min_value=0, max_value=50), max_size=6),
    budget=st.integers(min_value=1, max_value=150),
)
def test_mandatory_context_is_reserved_or_overflow_is_explicit(
    mandatory_tokens: list[int],
    budget: int,
) -> None:
    items = [
        _item(f"required-{index}", tokens, mandatory=True)
        for index, tokens in enumerate(mandatory_tokens)
    ]
    policy = OptimizationPolicy(max_input_tokens=budget)
    total = sum(mandatory_tokens)
    if total > budget:
        with pytest.raises(MandatoryContextOverflow):
            validate_contextual_budget(items, policy=policy)
    else:
        contextual = validate_contextual_budget(items, policy=policy)
        assert contextual.mandatory_tokens == total
        assert contextual.optional_budget == budget - total
