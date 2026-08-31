"""Deterministic sentence-level extractive compression."""

from __future__ import annotations

import re

from contextos.compression.base import CompressionResult
from contextos.embeddings import DeterministicEmbeddingProvider, EmbeddingProvider
from contextos.models import ContextItem
from contextos.scoring import relevance_scores
from contextos.tokenization import Tokenizer

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\r?\n+")
_EXACT_FACT = re.compile(
    r"(?:\b[+-]?\d+(?:[.,]\d+)*(?:%|[A-Za-z]+)?\b|https?://\S+|\b[A-Z]{2,}\w*\b)"
)
_TERM = re.compile(r"[A-Za-z0-9_.:/\\-]{3,}")


class ExtractiveCompressor:
    """Select relevant original sentences and restore their source order."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._embedding_provider = embedding_provider or DeterministicEmbeddingProvider()

    def compress(
        self,
        item: ContextItem,
        target_tokens: int,
        task: str,
    ) -> CompressionResult:
        """Extract whole sentences without rewriting their text."""
        original_tokens = self._tokenizer.count_tokens(item.content)
        sentences = [
            part.strip() for part in _SENTENCE_BOUNDARY.split(item.content) if part.strip()
        ]
        if not sentences:
            return self._failure(item, original_tokens, "no_sentences")

        probes = [
            item.model_copy(update={"id": f"{item.id}:sentence:{index}", "content": sentence})
            for index, sentence in enumerate(sentences)
        ]
        relevance = relevance_scores(task, probes, provider=self._embedding_provider)
        task_terms = {term.casefold() for term in _TERM.findall(task)}

        def rank(index: int) -> tuple[int, int, float, int]:
            sentence_terms = {term.casefold() for term in _TERM.findall(sentences[index])}
            return (
                -int(bool(_EXACT_FACT.search(sentences[index]))),
                -len(task_terms & sentence_terms),
                -relevance[probes[index].id],
                index,
            )

        ranked_indices = sorted(
            range(len(sentences)),
            key=rank,
        )
        selected: set[int] = set()
        for index in ranked_indices:
            proposed = " ".join(
                sentence
                for sentence_index, sentence in enumerate(sentences)
                if sentence_index in selected or sentence_index == index
            )
            if self._tokenizer.count_tokens(proposed) <= target_tokens:
                selected.add(index)

        content = " ".join(
            sentence for index, sentence in enumerate(sentences) if index in selected
        )
        if not content.strip():
            return self._failure(item, original_tokens, "no_sentence_fits_target")
        compressed_tokens = self._tokenizer.count_tokens(content)
        return CompressionResult(
            content=content,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            source_item_id=item.id,
            strategy="extractive",
            provenance=(item.id,),
            lossy=content != item.content,
        )

    @staticmethod
    def _failure(item: ContextItem, original_tokens: int, reason: str) -> CompressionResult:
        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=0,
            source_item_id=item.id,
            strategy="extractive",
            provenance=(item.id,),
            failure_reason=reason,
        )
