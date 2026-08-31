"""Embedding provider contracts and output validation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from contextos.errors import InvalidEmbeddingOutput

EmbeddingMatrix = NDArray[np.float64]


class EmbeddingProvider(Protocol):
    """Create one numeric embedding per supplied text."""

    def embed(self, texts: Sequence[str]) -> EmbeddingMatrix:
        """Return a two-dimensional matrix with one row per text."""
        ...


def validate_embedding_matrix(
    embeddings: object,
    *,
    expected_rows: int,
) -> EmbeddingMatrix:
    """Validate and normalize an embedding provider's public result."""
    try:
        matrix = np.asarray(embeddings, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidEmbeddingOutput("embedding output must be a numeric matrix") from exc
    if matrix.ndim != 2:
        raise InvalidEmbeddingOutput("embedding output must be two-dimensional")
    if matrix.shape[0] != expected_rows:
        raise InvalidEmbeddingOutput(
            f"embedding output has {matrix.shape[0]} rows; expected {expected_rows}"
        )
    if expected_rows and matrix.shape[1] == 0:
        raise InvalidEmbeddingOutput("embedding vectors must not be empty")
    if not np.isfinite(matrix).all():
        raise InvalidEmbeddingOutput("embedding output contains non-finite values")
    if expected_rows and np.any(np.linalg.norm(matrix, axis=1) == 0):
        raise InvalidEmbeddingOutput("embedding output contains a zero-length vector")
    return matrix
