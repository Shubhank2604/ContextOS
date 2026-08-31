"""Tests for deterministic, cached, and optional embedding providers."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import pytest

from contextos.embeddings import (
    CachedEmbeddingProvider,
    DeterministicEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from contextos.embeddings.base import EmbeddingMatrix, validate_embedding_matrix
from contextos.errors import EmbeddingProviderError, InvalidEmbeddingOutput


class CountingProvider:
    """Record cache misses while returning valid deterministic vectors."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> EmbeddingMatrix:
        self.calls.append(list(texts))
        return DeterministicEmbeddingProvider(dimensions=32).embed(texts)


def test_deterministic_embeddings_are_stable_and_normalized() -> None:
    provider = DeterministicEmbeddingProvider(dimensions=64)
    first = provider.embed(["Deploy after tests pass", "Different text"])
    second = provider.embed(["Deploy after tests pass", "Different text"])
    assert first.shape == (2, 64)
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)


def test_embedding_cache_normalizes_content_and_batches_only_misses() -> None:
    wrapped = CountingProvider()
    provider = CachedEmbeddingProvider(wrapped)
    first = provider.embed([" Café\r\n", "new value"])
    second = provider.embed(["Café\n", "new value"])
    assert np.array_equal(first, second)
    assert wrapped.calls == [[" Café\r\n", "new value"]]
    assert provider.cache_size == 2
    provider.clear()
    assert provider.cache_size == 0


@pytest.mark.parametrize(
    "value",
    [
        [1.0, 2.0],
        [[1.0], [2.0]],
        [[float("nan")]],
        [[0.0, 0.0]],
    ],
)
def test_malformed_embedding_output_is_rejected(value: object) -> None:
    with pytest.raises(InvalidEmbeddingOutput):
        validate_embedding_matrix(value, expected_rows=1)


def test_sentence_transformer_is_lazy_and_unavailable_provider_is_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = SentenceTransformerEmbeddingProvider("configured/model")

    def unavailable(_: str) -> object:
        raise ImportError("not installed")

    monkeypatch.setattr("contextos.embeddings.sentence_transformer.import_module", unavailable)
    with pytest.raises(EmbeddingProviderError, match="unavailable"):
        provider.embed(["text"])


def test_sentence_transformer_empty_batch_does_not_load_model() -> None:
    provider = SentenceTransformerEmbeddingProvider("configured/model")
    assert provider.embed([]).shape == (0, 0)


def test_sentence_transformer_uses_configured_model_and_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_names: list[str] = []

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            loaded_names.append(model_name)

        def encode(
            self,
            sentences: list[str],
            *,
            convert_to_numpy: bool,
            normalize_embeddings: bool,
        ) -> object:
            assert convert_to_numpy is True
            assert normalize_embeddings is True
            return np.ones((len(sentences), 3), dtype=np.float64)

    module = SimpleNamespace(SentenceTransformer=FakeModel)
    monkeypatch.setattr("contextos.embeddings.sentence_transformer.import_module", lambda _: module)
    provider = SentenceTransformerEmbeddingProvider("configured/model")
    assert provider.embed(["one", "two"]).shape == (2, 3)
    assert provider.embed(["three"]).shape == (1, 3)
    assert loaded_names == ["configured/model"]


def test_sentence_transformer_wraps_encode_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingModel:
        def __init__(self, _: str) -> None:
            pass

        def encode(self, *_: object, **__: object) -> object:
            raise RuntimeError("provider details")

    module = SimpleNamespace(SentenceTransformer=FailingModel)
    monkeypatch.setattr("contextos.embeddings.sentence_transformer.import_module", lambda _: module)
    provider = SentenceTransformerEmbeddingProvider("configured/model")
    with pytest.raises(EmbeddingProviderError, match="embedding failed"):
        provider.embed(["text"])


@pytest.mark.parametrize("dimensions", [0, -1, True])
def test_deterministic_provider_rejects_invalid_dimensions(dimensions: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        DeterministicEmbeddingProvider(dimensions)
