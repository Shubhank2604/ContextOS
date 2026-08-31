"""Tests for importance, recency, type priority, and composite scoring."""

from datetime import UTC, datetime, timedelta

import pytest

from contextos.config import DEFAULT_TYPE_PRIORITIES, OptimizationPolicy
from contextos.embeddings import DeterministicEmbeddingProvider
from contextos.errors import InvalidScore
from contextos.models import ContextEdge, ContextItem, ContextType, DependencyRelation
from contextos.scoring import (
    composite_scores,
    importance_scores,
    recency_scores,
    score_context_items,
    type_priority_scores,
)


def _item(
    item_id: str,
    *,
    context_type: ContextType = ContextType.MEMORY,
    updated_at: datetime | None = None,
    importance: float = 0.5,
    mandatory: bool = False,
) -> ContextItem:
    timestamp = updated_at or datetime(2026, 1, 1, tzinfo=UTC)
    return ContextItem(
        id=item_id,
        content=item_id,
        type=context_type,
        created_at=timestamp,
        updated_at=timestamp,
        importance=importance,
        mandatory=mandatory,
        evictable=not mandatory,
    )


def test_importance_is_application_supplied_and_type_priority_is_configurable() -> None:
    code = _item("code", context_type=ContextType.CODE, importance=0.33)
    assert importance_scores([code]) == {"code": 0.33}
    assert type_priority_scores([code]) == {"code": DEFAULT_TYPE_PRIORITIES[ContextType.CODE]}
    assert type_priority_scores([code], priorities={ContextType.CODE: 0.2}) == {"code": 0.2}


def test_recency_uses_exponential_half_life_and_equal_timestamps_match() -> None:
    now = datetime(2026, 1, 2, tzinfo=UTC)
    items = [
        _item("new-a", updated_at=now),
        _item("new-b", updated_at=now),
        _item("old", updated_at=now - timedelta(hours=1)),
    ]
    scores = recency_scores(items, half_life_seconds=3_600, reference_time=now)
    assert scores == {"new-a": 1.0, "new-b": 1.0, "old": 0.5}


def test_recency_default_reference_is_deterministic_and_future_age_is_clamped() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    items = [_item("old", updated_at=start), _item("new", updated_at=start + timedelta(hours=1))]
    assert recency_scores(items, half_life_seconds=3_600) == {"old": 0.5, "new": 1.0}
    assert recency_scores(
        [_item("future", updated_at=start + timedelta(hours=1))],
        half_life_seconds=3_600,
        reference_time=start,
    ) == {"future": 1.0}


def test_recency_rejects_invalid_configuration_and_accepts_empty_context() -> None:
    assert recency_scores([], half_life_seconds=1.0) == {}
    with pytest.raises(ValueError, match="positive"):
        recency_scores([], half_life_seconds=0.0)
    with pytest.raises(ValueError, match="timezone-aware"):
        recency_scores(
            [_item("item")],
            half_life_seconds=1.0,
            reference_time=datetime(2026, 1, 1),
        )


def test_composite_score_uses_normalized_weights_and_preserves_mandatory_flag() -> None:
    item = _item("required", mandatory=True)
    policy = OptimizationPolicy(
        max_input_tokens=100,
        weight_relevance=1,
        weight_importance=1,
        weight_recency=1,
        weight_novelty=1,
        weight_dependency=1,
        weight_type_priority=1,
    )
    result = composite_scores(
        [item],
        policy=policy,
        relevance={item.id: 1.0},
        importance={item.id: 0.8},
        recency={item.id: 0.6},
        novelty={item.id: 0.4},
        dependency={item.id: 0.2},
        type_priority={item.id: 0.0},
    )[item.id]
    assert result.composite_utility == pytest.approx(0.5)
    assert item.mandatory is True
    assert item.evictable is False


@pytest.mark.parametrize(
    "bad_component",
    ["relevance", "importance", "recency", "novelty", "dependency", "type_priority"],
)
def test_composite_rejects_missing_or_out_of_range_component(bad_component: str) -> None:
    item = _item("item")
    components: dict[str, dict[str, float]] = {
        "relevance": {"item": 0.5},
        "importance": {"item": 0.5},
        "recency": {"item": 0.5},
        "novelty": {"item": 0.5},
        "dependency": {"item": 0.5},
        "type_priority": {"item": 0.5},
    }
    components[bad_component] = {"item": 1.1}
    with pytest.raises(InvalidScore, match=bad_component):
        composite_scores([item], policy=OptimizationPolicy(max_input_tokens=10), **components)


def test_composite_rejects_missing_component() -> None:
    item = _item("item")
    with pytest.raises(InvalidScore, match="relevance score missing"):
        composite_scores(
            [item],
            policy=OptimizationPolicy(max_input_tokens=10),
            relevance={},
            importance={"item": 0.5},
            recency={"item": 0.5},
            novelty={"item": 0.5},
            dependency={"item": 0.5},
            type_priority={"item": 0.5},
        )


def test_full_scoring_pipeline_is_deterministic_and_dependency_aware() -> None:
    now = datetime(2026, 1, 2, tzinfo=UTC)
    items = [
        _item("primary", updated_at=now, importance=1.0),
        _item("dependency", updated_at=now - timedelta(hours=1), importance=0.1),
        _item("required", updated_at=now, mandatory=True),
    ]
    edge = ContextEdge(
        source_id="primary",
        target_id="dependency",
        relation=DependencyRelation.REQUIRES,
        weight=0.8,
    )
    policy = OptimizationPolicy.balanced(max_input_tokens=100)
    provider = DeterministicEmbeddingProvider()
    first = score_context_items(
        "primary task",
        items,
        edges=[edge],
        policy=policy,
        provider=provider,
        reference_time=now,
    )
    second = score_context_items(
        "primary task",
        items,
        edges=[edge],
        policy=policy,
        provider=provider,
        reference_time=now,
    )
    assert first == second
    assert first["dependency"].dependency > 0.0
    assert first["required"].novelty == 1.0
    assert items[2].mandatory is True


def test_scoring_pipeline_can_disable_semantic_relevance() -> None:
    item = _item("item")
    result = score_context_items(
        "task",
        [item],
        edges=[],
        policy=OptimizationPolicy(max_input_tokens=10, semantic_relevance_enabled=False),
        provider=DeterministicEmbeddingProvider(),
    )
    assert result["item"].relevance == 0.0
