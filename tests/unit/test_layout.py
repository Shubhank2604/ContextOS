"""Tests for independent deterministic layout strategies."""

from datetime import UTC, datetime, timedelta

import pytest

from contextos.errors import InvalidScore
from contextos.layout import (
    OriginalOrderLayout,
    PositionAwareLayout,
    RelevanceDescendingLayout,
)
from contextos.models import ContextItem, ContextType
from contextos.scoring import ScoreBreakdown

LayoutInputs = tuple[list[ContextItem], dict[str, ScoreBreakdown], dict[str, int]]


def make_item(
    item_id: str,
    context_type: ContextType,
    position: int,
    *,
    mandatory: bool = False,
) -> ContextItem:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=position)
    return ContextItem(
        id=item_id,
        content="line one\nline two" if context_type is ContextType.CODE else item_id,
        type=context_type,
        created_at=timestamp,
        updated_at=timestamp,
        mandatory=mandatory,
        evictable=not mandatory,
    )


def score(relevance: float, utility: float) -> ScoreBreakdown:
    return ScoreBreakdown(
        relevance=relevance,
        importance=0.5,
        recency=0.5,
        novelty=0.5,
        dependency=0.5,
        type_priority=0.5,
        composite_utility=utility,
    )


@pytest.fixture
def layout_inputs() -> LayoutInputs:
    items = [
        make_item("old-state", ContextType.TASK_STATE, 0),
        make_item("code", ContextType.CODE, 1),
        make_item("system", ContextType.SYSTEM_INSTRUCTION, 2, mandatory=True),
        make_item("evidence", ContextType.RETRIEVED_DOCUMENT, 3),
        make_item("user", ContextType.USER_MESSAGE, 4),
    ]
    scores = {
        "old-state": score(0.4, 0.8),
        "code": score(0.7, 0.6),
        "system": score(1.0, 1.0),
        "evidence": score(0.9, 0.9),
        "user": score(0.8, 0.7),
    }
    positions = {item.id: index for index, item in enumerate(items)}
    return items, scores, positions


def test_original_order_layout_restores_input_order(
    layout_inputs: LayoutInputs,
) -> None:
    items, scores, positions = layout_inputs
    arranged = OriginalOrderLayout().arrange(
        list(reversed(items)), scores=scores, original_positions=positions
    )
    assert [item.id for item in arranged] == [item.id for item in items]


def test_relevance_layout_is_descending(layout_inputs: LayoutInputs) -> None:
    items, scores, positions = layout_inputs
    arranged = RelevanceDescendingLayout().arrange(
        items, scores=scores, original_positions=positions
    )
    assert [item.id for item in arranged] == ["system", "evidence", "user", "code", "old-state"]


def test_position_aware_regions_and_code_content_are_stable(
    layout_inputs: LayoutInputs,
) -> None:
    items, scores, positions = layout_inputs
    arranged = PositionAwareLayout().arrange(items, scores=scores, original_positions=positions)

    assert [item.id for item in arranged] == ["system", "evidence", "code", "old-state", "user"]
    assert next(item for item in arranged if item.id == "code").content == "line one\nline two"
    second = PositionAwareLayout().arrange(items, scores=scores, original_positions=positions)
    assert [item.id for item in second] == [item.id for item in arranged]


def test_layout_rejects_missing_scores(layout_inputs: LayoutInputs) -> None:
    items, scores, positions = layout_inputs
    del scores["code"]
    with pytest.raises(InvalidScore, match="code"):
        PositionAwareLayout().arrange(items, scores=scores, original_positions=positions)
