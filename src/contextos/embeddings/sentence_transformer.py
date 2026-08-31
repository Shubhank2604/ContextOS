"""Optional sentence-transformers embedding provider."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Protocol, cast

import numpy as np

from contextos.config import DEFAULT_SENTENCE_TRANSFORMER_MODEL
from contextos.embeddings.base import EmbeddingMatrix, validate_embedding_matrix
from contextos.errors import EmbeddingProviderError


class _SentenceTransformerModel(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> object: ...


class SentenceTransformerEmbeddingProvider:
    """Lazily load a configurable local sentence-transformers model."""

    def __init__(self, model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        self.model_name = model_name
        self._model: _SentenceTransformerModel | None = None

    def _load_model(self) -> _SentenceTransformerModel:
        if self._model is not None:
            return self._model
        try:
            module = import_module("sentence_transformers")
            model_type = module.SentenceTransformer
            model = model_type(self.model_name)
        except Exception as exc:
            raise EmbeddingProviderError(
                "sentence-transformers is unavailable or the configured model could not be loaded"
            ) from exc
        self._model = cast(_SentenceTransformerModel, model)
        return self._model

    def embed(self, texts: Sequence[str]) -> EmbeddingMatrix:
        """Embed a batch, loading the optional model only on first use."""
        if not texts:
            return np.empty((0, 0), dtype=np.float64)
        try:
            embeddings = self._load_model().encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except EmbeddingProviderError:
            raise
        except Exception as exc:
            raise EmbeddingProviderError("sentence-transformer embedding failed") from exc
        return validate_embedding_matrix(embeddings, expected_rows=len(texts))
