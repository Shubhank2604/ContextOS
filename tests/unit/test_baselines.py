"""Tests for deterministic baseline strategies and traces."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from contextos.baselines import (
    BaselineStrategy,
    FullContextBaseline,
    LastNTokensBaseline,
    NaiveExtractiveBaseline,
    RelevanceOnlyBaseline,
    SlidingWindowBaseline,
)
from contextos.config import OptimizationPolicy
from contextos.errors import (
    ContextBudgetOverflow,
    DuplicateContextItemError,
    InvalidOptimizationPolicy,
)
from contextos.models import ContextItem, ContextType
from contextos.trace import OptimizationDecision, OptimizedContext


class WordTokenizer:
    """Deterministic whole-word tokenizer for selection tests."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class FailingTokenizer:
    """Tokenizer that proves invalid policies fail before tokenization."""

    def count_tokens(self, text: str) -> int:
        raise AssertionError(f"tokenizer must not be called for {text!r}")


def make_item(
    item_id: str,
    content: str,
    age_hours: int,
    *,
    mandatory: bool = False,
) -> ContextItem:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=age_hours)
    return ContextItem(
        id=item_id,
        content=content,
        type=ContextType.MEMORY,
        created_at=timestamp,
        updated_at=timestamp,
        importance=0.5,
        mandatory=mandatory,
        evictable=not mandatory,
    )


def optimize(strategy: BaselineStrategy, items: list[ContextItem], budget: int) -> OptimizedContext:
    return strategy.optimize(
        task="test task",
        items=items,
        policy=OptimizationPolicy(max_input_tokens=budget),
        tokenizer=WordTokenizer(),
    )


def test_full_context_retains_input_order_and_traces_every_item() -> None:
    items = [make_item("old", "one two", 0), make_item("new", "three", 1)]
    result = optimize(FullContextBaseline(), items, budget=3)

    assert [item.id for item in result.selected_items] == ["old", "new"]
    assert result.removed_items == []
    assert result.final_token_count == 3
    assert result.trace.selected_count == 2
    assert [item.final_position for item in result.trace.items] == [0, 1]
    assert all(item.decision is OptimizationDecision.RETAINED for item in result.trace.items)


def test_full_context_overflow_is_explicit() -> None:
    with pytest.raises(ContextBudgetOverflow) as raised:
        optimize(FullContextBaseline(), [make_item("large", "one two three", 0)], budget=2)
    assert raised.value.strategy == "full_context"
    assert raised.value.required_tokens == 3
    assert raised.value.effective_budget == 2


def test_invalid_policy_fails_before_tokenization() -> None:
    with pytest.raises(InvalidOptimizationPolicy):
        FullContextBaseline().optimize(
            task="test",
            items=[make_item("item", "one", 0)],
            policy=OptimizationPolicy(max_input_tokens=1, reserve_output_tokens=1),
            tokenizer=FailingTokenizer(),
        )


def test_last_n_retains_newest_contiguous_whole_item_suffix() -> None:
    items = [
        make_item("old", "one", 0),
        make_item("middle", "two three", 1),
        make_item("new", "four", 2),
    ]
    result = optimize(LastNTokensBaseline(), items, budget=3)

    assert [item.id for item in result.selected_items] == ["middle", "new"]
    assert [item.id for item in result.removed_items] == ["old"]
    assert result.final_token_count <= result.budget_allocation.effective_budget
    assert result.trace.items[0].decision_reason == "outside_last_n_budget"


def test_last_n_stops_when_newest_whole_item_does_not_fit() -> None:
    items = [make_item("old-small", "one", 0), make_item("new-large", "one two three", 1)]
    result = optimize(LastNTokensBaseline(), items, budget=2)
    assert result.selected_items == []
    assert result.budget_allocation.remaining_tokens == 2


def test_baseline_mandatory_limitation_is_explicit_in_trace() -> None:
    items = [make_item("old-required", "must stay", 0, mandatory=True)]
    result = optimize(LastNTokensBaseline(), items, budget=1)
    assert result.selected_items == []
    assert result.trace.mandatory_tokens == 2
    assert result.trace.optional_budget == 0
    assert result.trace.warnings == ["baseline_does_not_enforce_mandatory_retention"]


def test_sliding_window_excludes_old_items() -> None:
    items = [
        make_item("old", "one", 0),
        make_item("recent", "two", 3),
        make_item("new", "three", 4),
    ]
    result = optimize(SlidingWindowBaseline(window_seconds=2 * 60 * 60), items, budget=5)
    assert [item.id for item in result.selected_items] == ["recent", "new"]
    assert result.trace.items[0].decision_reason == "outside_recent_window"


def test_sliding_window_applies_newest_budget_within_window() -> None:
    items = [
        make_item("window-old", "one two", 2),
        make_item("window-new", "three four", 3),
    ]
    result = optimize(SlidingWindowBaseline(window_seconds=2 * 60 * 60), items, budget=2)
    assert [item.id for item in result.selected_items] == ["window-new"]
    assert result.trace.items[0].decision_reason == "recent_window_budget_exhausted"


def test_relevance_only_selects_task_matching_whole_items() -> None:
    items = [
        make_item("unrelated", "formatting colors", 0),
        make_item("relevant", "authentication timeout", 1),
    ]
    result = RelevanceOnlyBaseline().optimize(
        task="authentication timeout",
        items=items,
        policy=OptimizationPolicy(max_input_tokens=2),
        tokenizer=WordTokenizer(),
    )

    assert [item.id for item in result.selected_items] == ["relevant"]
    assert result.trace.items[1].relevance_score == pytest.approx(1.0)
    assert result.trace.items[0].decision_reason == "outside_relevance_budget"


def test_naive_extractive_selects_relevant_sentences_under_budget() -> None:
    item = make_item(
        "mixed",
        "Formatting colors changed. Authentication timeout failed.",
        0,
    )
    result = NaiveExtractiveBaseline().optimize(
        task="authentication timeout",
        items=[item],
        policy=OptimizationPolicy(max_input_tokens=3),
        tokenizer=WordTokenizer(),
    )

    assert result.selected_items[0].content == "Authentication timeout failed."
    assert result.final_token_count == 3
    assert result.trace.compressed_count == 1
    assert result.trace.items[0].decision is OptimizationDecision.COMPRESSED
    assert result.trace.items[0].compression_strategy == "naive_extractive"


@pytest.mark.parametrize("window_seconds", [0, -1, True])
def test_sliding_window_requires_positive_integer(window_seconds: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        SlidingWindowBaseline(window_seconds)


def test_baselines_do_not_mutate_source_items() -> None:
    item = make_item("source", "one two", 0)
    for strategy in (
        LastNTokensBaseline(),
        RelevanceOnlyBaseline(),
        NaiveExtractiveBaseline(),
    ):
        optimize(strategy, [item], budget=2)
    assert item.token_count is None


def test_duplicate_ids_fail_before_selection() -> None:
    item = make_item("duplicate", "one", 0)
    with pytest.raises(DuplicateContextItemError):
        optimize(LastNTokensBaseline(), [item, item.model_copy(deep=True)], budget=2)


def test_empty_context_has_complete_zero_trace() -> None:
    result = optimize(FullContextBaseline(), [], budget=4)
    assert result.original_token_count == 0
    assert result.final_token_count == 0
    assert result.trace.reduction_ratio == 0.0
    assert result.trace.items == []
