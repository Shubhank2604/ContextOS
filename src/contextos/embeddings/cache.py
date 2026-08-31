"""Content-addressed in-memory embedding cache."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from contextos.dedup.exact import normalized_content_hash
from contextos.embeddings.base import (
    EmbeddingMatrix,
    EmbeddingProvider,
    validate_embedding_matrix,
)


class CachedEmbeddingProvider:
    """Cache provider vectors by normalized SHA-256 content hash."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider
        self._cache: dict[str, EmbeddingMatrix] = {}

    @property
    def cache_size(self) -> int:
        """Return the number of distinct normalized texts cached."""
        return len(self._cache)

    def clear(self) -> None:
        """Discard all cached vectors."""
        self._cache.clear()

    def embed(self, texts: Sequence[str]) -> EmbeddingMatrix:
        """Embed only cache misses and restore the caller's row order."""
        if not texts:
            return np.empty((0, 0), dtype=np.float64)

        keys = [normalized_content_hash(text) for text in texts]
        missing: dict[str, str] = {}
        for key, value in zip(keys, texts, strict=True):
            if key not in self._cache and key not in missing:
                missing[key] = value

        if missing:
            missing_keys = list(missing)
            matrix = validate_embedding_matrix(
                self.provider.embed(list(missing.values())),
                expected_rows=len(missing),
            )
            for row, key in enumerate(missing_keys):
                self._cache[key] = matrix[row : row + 1].copy()

        rows = [self._cache[key] for key in keys]
        dimensions = {row.shape[1] for row in rows}
        if len(dimensions) != 1:
            raise ValueError("cached embedding dimensions changed between provider calls")
        return np.concatenate(rows, axis=0)
