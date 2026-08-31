"""Nearest-neighbor novelty scoring for optional context."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from contextos.dedup.semantic import cosine_similarity
from contextos.embeddings.base import EmbeddingProvider, validate_embedding_matrix
from contextos.models import ContextItem


def novelty_scores(
    items: Sequence[ContextItem],
    *,
    provider: EmbeddingProvider,
) -> dict[str, float]:
    """Return ``1 - nearest normalized cosine`` for every optional item."""
    optional = [item for item in items if not item.mandatory]
    if not optional:
        return {}
    if len(optional) == 1:
        return {optional[0].id: 1.0}
    matrix = validate_embedding_matrix(
        provider.embed([item.content for item in optional]),
        expected_rows=len(optional),
    )
    scores: dict[str, float] = {}
    for left_index, item in enumerate(optional):
        nearest = max(
            (cosine_similarity(matrix[left_index], matrix[right_index]) + 1) / 2
            for right_index in range(len(optional))
            if right_index != left_index
        )
        scores[item.id] = float(np.clip(1 - nearest, 0, 1))
    return scores
