"""Normalized weighted composite-utility scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from contextos.config import OptimizationPolicy
from contextos.errors import InvalidScore
from contextos.models import ContextItem


class ScoreBreakdown(BaseModel):
    """All normalized scoring evidence for one item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relevance: float = Field(ge=0.0, le=1.0)
    importance: float = Field(ge=0.0, le=1.0)
    recency: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    dependency: float = Field(ge=0.0, le=1.0)
    type_priority: float = Field(ge=0.0, le=1.0)
    composite_utility: float = Field(ge=0.0, le=1.0)


def _score_for(component: str, item_id: str, scores: Mapping[str, float]) -> float:
    try:
        score = scores[item_id]
    except KeyError as exc:
        raise InvalidScore(f"{component} score missing for item {item_id}") from exc
    if not 0.0 <= score <= 1.0:
        raise InvalidScore(f"{component} score for {item_id} must be between 0 and 1")
    return score


def composite_scores(
    items: Sequence[ContextItem],
    *,
    policy: OptimizationPolicy,
    relevance: Mapping[str, float],
    importance: Mapping[str, float],
    recency: Mapping[str, float],
    novelty: Mapping[str, float],
    dependency: Mapping[str, float],
    type_priority: Mapping[str, float],
) -> dict[str, ScoreBreakdown]:
    """Combine every component using statically validated normalized weights."""
    weights = policy.normalized_weights
    results: dict[str, ScoreBreakdown] = {}
    for item in items:
        values = {
            "relevance": _score_for("relevance", item.id, relevance),
            "importance": _score_for("importance", item.id, importance),
            "recency": _score_for("recency", item.id, recency),
            "novelty": _score_for("novelty", item.id, novelty),
            "dependency": _score_for("dependency", item.id, dependency),
            "type_priority": _score_for("type_priority", item.id, type_priority),
        }
        utility = sum(weights[name] * score for name, score in values.items())
        results[item.id] = ScoreBreakdown(**values, composite_utility=min(max(utility, 0.0), 1.0))
    return results
