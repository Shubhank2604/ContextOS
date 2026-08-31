"""Deterministic labeled-fixture metrics for deduplication quality."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from contextos.dedup.exact import exact_deduplicate
from contextos.dedup.semantic import semantic_deduplicate
from contextos.embeddings.base import EmbeddingProvider
from contextos.models import ContextItem, ContextType


class DeduplicationCase(BaseModel):
    """One labeled duplicate/non-duplicate content pair."""

    model_config = ConfigDict(extra="forbid")

    id: str
    left: str
    right: str
    context_type: ContextType
    should_deduplicate: bool


class DeduplicationMetrics(BaseModel):
    """Raw confusion counts plus derived duplicate-class metrics."""

    model_config = ConfigDict(extra="forbid")

    case_count: int = Field(ge=0)
    true_positive: int = Field(ge=0)
    true_negative: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)


def evaluate_deduplication_cases(
    cases: list[DeduplicationCase],
    *,
    provider: EmbeddingProvider,
    threshold: float = 0.92,
) -> DeduplicationMetrics:
    """Evaluate exact-plus-semantic deduplication on labeled pairs."""
    true_positive = true_negative = false_positive = false_negative = 0
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    for case in cases:
        items = [
            ContextItem(
                id=f"{case.id}-left",
                content=case.left,
                type=case.context_type,
                created_at=timestamp,
                updated_at=timestamp,
                importance=0.5,
            ),
            ContextItem(
                id=f"{case.id}-right",
                content=case.right,
                type=case.context_type,
                created_at=timestamp,
                updated_at=timestamp,
                importance=0.5,
            ),
        ]
        exact = exact_deduplicate(items)
        semantic = semantic_deduplicate(exact.items, provider=provider, threshold=threshold)
        predicted_duplicate = len(semantic.items) < 2 or len(exact.items) < 2
        if predicted_duplicate and case.should_deduplicate:
            true_positive += 1
        elif predicted_duplicate:
            false_positive += 1
        elif case.should_deduplicate:
            false_negative += 1
        else:
            true_negative += 1

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return DeduplicationMetrics(
        case_count=len(cases),
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
    )
