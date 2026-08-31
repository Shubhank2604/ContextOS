"""Tests for relevance and novelty scores."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np

from contextos.embeddings.base import EmbeddingMatrix
from contextos.models import ContextItem, ContextType
from contextos.scoring import novelty_scores, relevance_scores


class OrderedProvider:
    """Return a fixed prefix of vectors for controlled cosine tests."""

    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows

    def embed(self, texts: Sequence[str]) -> EmbeddingMatrix:
        return np.asarray(self.rows[: len(texts)], dtype=np.float64)


def _item(item_id: str, *, mandatory: bool = False) -> ContextItem:
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
    )


def test_relevance_normalizes_cosine_to_score_bounds() -> None:
    scores = relevance_scores(
        "task",
        [_item("same"), _item("opposite"), _item("orthogonal")],
        provider=OrderedProvider([[1, 0], [1, 0], [-1, 0], [0, 1]]),
    )
    assert scores == {"same": 1.0, "opposite": 0.0, "orthogonal": 0.5}


def test_novelty_uses_nearest_optional_neighbor_and_skips_mandatory() -> None:
    scores = novelty_scores(
        [_item("a"), _item("b"), _item("mandatory", mandatory=True)],
        provider=OrderedProvider([[1, 0], [0, 1]]),
    )
    assert scores == {"a": 0.5, "b": 0.5}


def test_single_optional_item_is_fully_novel_without_embedding() -> None:
    assert novelty_scores([_item("only")], provider=OrderedProvider([])) == {"only": 1.0}


def test_empty_context_has_no_semantic_scores() -> None:
    provider = OrderedProvider([])
    assert relevance_scores("task", [], provider=provider) == {}
    assert novelty_scores([_item("mandatory", mandatory=True)], provider=provider) == {}
