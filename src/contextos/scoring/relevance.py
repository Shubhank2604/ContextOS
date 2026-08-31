"""Task-to-context semantic relevance scoring."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from contextos.dedup.semantic import cosine_similarity
from contextos.embeddings.base import EmbeddingProvider, validate_embedding_matrix
from contextos.models import ContextItem


def relevance_scores(
    task: str,
    items: Sequence[ContextItem],
    *,
    provider: EmbeddingProvider,
) -> dict[str, float]:
    """Return cosine-derived relevance scores normalized into ``[0, 1]``."""
    if not items:
        return {}
    matrix = validate_embedding_matrix(
        provider.embed([task, *(item.content for item in items)]),
        expected_rows=len(items) + 1,
    )
    task_embedding = matrix[0]
    return {
        item.id: float(np.clip((cosine_similarity(task_embedding, matrix[index]) + 1) / 2, 0, 1))
        for index, item in enumerate(items, start=1)
    }
