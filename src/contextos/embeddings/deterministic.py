"""Deterministic feature-hash embeddings for tests and local smoke runs."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Sequence

import numpy as np

from contextos.embeddings.base import EmbeddingMatrix

_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[_'-][^\W_]+)*", re.UNICODE)


class DeterministicEmbeddingProvider:
    """Map lexical features into a stable, normalized hash-vector space."""

    model_name = "contextos-deterministic-feature-hash-v1"

    def __init__(self, dimensions: int = 1024) -> None:
        if type(dimensions) is not int or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> EmbeddingMatrix:
        """Return deterministic L2-normalized bag-of-token vectors."""
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float64)
        for row, text in enumerate(texts):
            normalized = unicodedata.normalize("NFKC", text).casefold()
            tokens = _TOKEN_PATTERN.findall(normalized)
            if not tokens:
                tokens = [normalized or "<empty>"]
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                column = int.from_bytes(digest[:8], "big") % self.dimensions
                matrix[row, column] += 1.0
            norm = float(np.linalg.norm(matrix[row]))
            matrix[row] /= norm
        return matrix
