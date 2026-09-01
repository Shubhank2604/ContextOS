"""Deterministic baseline context-selection strategies."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from time import perf_counter
from typing import Protocol

from contextos.config import OptimizationPolicy
from contextos.embeddings import DeterministicEmbeddingProvider, EmbeddingProvider
from contextos.errors import ContextBudgetOverflow
from contextos.models import ContextItem, validate_unique_item_ids
from contextos.scoring import relevance_scores
from contextos.tokenization import Tokenizer
from contextos.trace import (
    BudgetAllocation,
    ItemTrace,
    OptimizationDecision,
    OptimizationTrace,
    OptimizedContext,
)


class BaselineStrategy(Protocol):
    """Construct a baseline context under a token-budget policy."""

    name: str

    def optimize(
        self,
        *,
        task: str,
        items: Sequence[ContextItem],
        policy: OptimizationPolicy,
        tokenizer: Tokenizer,
    ) -> OptimizedContext:
        """Return the selected baseline context and complete trace."""
        ...


@dataclass(frozen=True)
class _PreparedItem:
    index: int
    item: ContextItem


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\r?\n+")


def _prepare_items(
    items: Sequence[ContextItem], policy: OptimizationPolicy, tokenizer: Tokenizer
) -> tuple[list[_PreparedItem], dict[str, float]]:
    started = perf_counter()
    policy.validate_static()
    validate_unique_item_ids(items)
    validated_at = perf_counter()

    prepared: list[_PreparedItem] = []
    for index, original in enumerate(items):
        item = original.model_copy(deep=True)
        item.token_count = tokenizer.count_tokens(item.content)
        prepared.append(_PreparedItem(index=index, item=item))
    tokenized_at = perf_counter()
    return prepared, {
        "validate": (validated_at - started) * 1000,
        "tokenize": (tokenized_at - validated_at) * 1000,
    }


def _newest_first(prepared: Sequence[_PreparedItem]) -> list[_PreparedItem]:
    return sorted(
        prepared,
        key=lambda value: (
            -value.item.updated_at.timestamp(),
            -value.item.created_at.timestamp(),
            value.item.id,
        ),
    )


def _select_contiguous_newest(prepared: Sequence[_PreparedItem], effective_budget: int) -> set[str]:
    selected: set[str] = set()
    used_tokens = 0
    for candidate in _newest_first(prepared):
        token_count = candidate.item.token_count or 0
        if used_tokens + token_count > effective_budget:
            break
        selected.add(candidate.item.id)
        used_tokens += token_count
    return selected


def _build_result(
    *,
    strategy: str,
    prepared: Sequence[_PreparedItem],
    selected_ids: set[str],
    removal_reasons: dict[str, str],
    policy: OptimizationPolicy,
    timings: dict[str, float],
    selection_started: float,
    relevance: dict[str, float] | None = None,
) -> OptimizedContext:
    selection_finished = perf_counter()
    ordered = sorted(prepared, key=lambda value: value.index)
    selected = [value.item for value in ordered if value.item.id in selected_ids]
    removed = [value.item for value in ordered if value.item.id not in selected_ids]
    final_positions = {item.id: position for position, item in enumerate(selected)}

    item_traces = [
        ItemTrace(
            item_id=value.item.id,
            initial_token_count=value.item.token_count or 0,
            decision=(
                OptimizationDecision.RETAINED
                if value.item.id in selected_ids
                else OptimizationDecision.REMOVED
            ),
            decision_reason=(
                "selected_by_baseline"
                if value.item.id in selected_ids
                else removal_reasons[value.item.id]
            ),
            final_token_count=(value.item.token_count or 0) if value.item.id in selected_ids else 0,
            final_position=final_positions.get(value.item.id),
            relevance_score=(relevance or {}).get(value.item.id),
            provenance=[value.item.id],
        )
        for value in ordered
    ]
    traced_at = perf_counter()

    original_tokens = sum(value.item.token_count or 0 for value in prepared)
    final_tokens = sum(item.token_count or 0 for item in selected)
    mandatory_tokens = sum(
        value.item.token_count or 0 for value in prepared if value.item.mandatory
    )
    warnings = ["baseline_does_not_enforce_mandatory_retention"] if mandatory_tokens else []
    effective_budget = policy.effective_budget
    reduction_ratio = 0.0 if original_tokens == 0 else 1 - (final_tokens / original_tokens)
    stage_timings = {
        **timings,
        "selection": (selection_finished - selection_started) * 1000,
        "trace": (traced_at - selection_finished) * 1000,
    }
    trace = OptimizationTrace(
        strategy=strategy,
        policy=policy,
        effective_budget=effective_budget,
        mandatory_tokens=mandatory_tokens,
        optional_budget=max(effective_budget - mandatory_tokens, 0),
        original_tokens=original_tokens,
        final_tokens=final_tokens,
        reduction_ratio=reduction_ratio,
        stage_timings_ms=stage_timings,
        selected_count=len(selected),
        removed_count=len(removed),
        compressed_count=0,
        warnings=warnings,
        items=item_traces,
    )
    return OptimizedContext(
        selected_items=selected,
        removed_items=removed,
        original_token_count=original_tokens,
        final_token_count=final_tokens,
        budget_allocation=BudgetAllocation(
            effective_budget=effective_budget,
            used_tokens=final_tokens,
            remaining_tokens=effective_budget - final_tokens,
        ),
        trace=trace,
    )


class FullContextBaseline:
    """Retain every whole item and fail explicitly when all items do not fit."""

    name = "full_context"

    def optimize(
        self,
        *,
        task: str,
        items: Sequence[ContextItem],
        policy: OptimizationPolicy,
        tokenizer: Tokenizer,
    ) -> OptimizedContext:
        del task
        prepared, timings = _prepare_items(items, policy, tokenizer)
        selection_started = perf_counter()
        original_tokens = sum(value.item.token_count or 0 for value in prepared)
        if original_tokens > policy.effective_budget:
            raise ContextBudgetOverflow(
                strategy=self.name,
                required_tokens=original_tokens,
                effective_budget=policy.effective_budget,
            )
        selected_ids = {value.item.id for value in prepared}
        return _build_result(
            strategy=self.name,
            prepared=prepared,
            selected_ids=selected_ids,
            removal_reasons={},
            policy=policy,
            timings=timings,
            selection_started=selection_started,
        )


class LastNTokensBaseline:
    """Retain the newest contiguous suffix of whole context items that fits."""

    name = "last_n"

    def optimize(
        self,
        *,
        task: str,
        items: Sequence[ContextItem],
        policy: OptimizationPolicy,
        tokenizer: Tokenizer,
    ) -> OptimizedContext:
        del task
        prepared, timings = _prepare_items(items, policy, tokenizer)
        selection_started = perf_counter()
        selected_ids = _select_contiguous_newest(prepared, policy.effective_budget)
        removal_reasons = {
            value.item.id: "outside_last_n_budget"
            for value in prepared
            if value.item.id not in selected_ids
        }
        return _build_result(
            strategy=self.name,
            prepared=prepared,
            selected_ids=selected_ids,
            removal_reasons=removal_reasons,
            policy=policy,
            timings=timings,
            selection_started=selection_started,
        )


class SlidingWindowBaseline:
    """Retain whole items in a recent time window, newest-first when over budget."""

    name = "sliding_window"

    def __init__(self, window_seconds: int) -> None:
        if type(window_seconds) is not int or window_seconds <= 0:
            raise ValueError("window_seconds must be a positive integer")
        self.window_seconds = window_seconds

    def optimize(
        self,
        *,
        task: str,
        items: Sequence[ContextItem],
        policy: OptimizationPolicy,
        tokenizer: Tokenizer,
    ) -> OptimizedContext:
        del task
        prepared, timings = _prepare_items(items, policy, tokenizer)
        selection_started = perf_counter()
        if prepared:
            reference_time = max(value.item.updated_at for value in prepared)
            cutoff = reference_time - timedelta(seconds=self.window_seconds)
            in_window = [value for value in prepared if value.item.updated_at >= cutoff]
        else:
            in_window = []
        selected_ids = _select_contiguous_newest(in_window, policy.effective_budget)
        in_window_ids = {value.item.id for value in in_window}
        removal_reasons = {
            value.item.id: (
                "outside_recent_window"
                if value.item.id not in in_window_ids
                else "recent_window_budget_exhausted"
            )
            for value in prepared
            if value.item.id not in selected_ids
        }
        return _build_result(
            strategy=self.name,
            prepared=prepared,
            selected_ids=selected_ids,
            removal_reasons=removal_reasons,
            policy=policy,
            timings=timings,
            selection_started=selection_started,
        )


class RelevanceOnlyBaseline:
    """Select whole items by task relevance without other ContextOS signals."""

    name = "relevance_only"

    def __init__(self, embedding_provider: EmbeddingProvider | None = None) -> None:
        self._embedding_provider = embedding_provider or DeterministicEmbeddingProvider()

    def optimize(
        self,
        *,
        task: str,
        items: Sequence[ContextItem],
        policy: OptimizationPolicy,
        tokenizer: Tokenizer,
    ) -> OptimizedContext:
        prepared, timings = _prepare_items(items, policy, tokenizer)
        relevance_started = perf_counter()
        relevance = relevance_scores(
            task,
            [value.item for value in prepared],
            provider=self._embedding_provider,
        )
        selection_started = perf_counter()
        timings["relevance"] = (selection_started - relevance_started) * 1000
        selected_ids: set[str] = set()
        used_tokens = 0
        for candidate in sorted(
            prepared,
            key=lambda value: (-relevance[value.item.id], value.index, value.item.id),
        ):
            token_count = candidate.item.token_count or 0
            if used_tokens + token_count <= policy.effective_budget:
                selected_ids.add(candidate.item.id)
                used_tokens += token_count
        removal_reasons = {
            value.item.id: "outside_relevance_budget"
            for value in prepared
            if value.item.id not in selected_ids
        }
        return _build_result(
            strategy=self.name,
            prepared=prepared,
            selected_ids=selected_ids,
            removal_reasons=removal_reasons,
            policy=policy,
            timings=timings,
            selection_started=selection_started,
            relevance=relevance,
        )


@dataclass(frozen=True)
class _SentenceCandidate:
    item_index: int
    sentence_index: int
    probe: ContextItem


class NaiveExtractiveBaseline:
    """Greedily retain task-relevant source sentences without ContextOS scoring."""

    name = "naive_extractive"

    def __init__(self, embedding_provider: EmbeddingProvider | None = None) -> None:
        self._embedding_provider = embedding_provider or DeterministicEmbeddingProvider()

    def optimize(
        self,
        *,
        task: str,
        items: Sequence[ContextItem],
        policy: OptimizationPolicy,
        tokenizer: Tokenizer,
    ) -> OptimizedContext:
        prepared, timings = _prepare_items(items, policy, tokenizer)
        candidates: list[_SentenceCandidate] = []
        sentences_by_item: dict[int, list[str]] = {}
        for prepared_item in prepared:
            sentences = [
                sentence.strip()
                for sentence in _SENTENCE_BOUNDARY.split(prepared_item.item.content)
                if sentence.strip()
            ]
            sentences_by_item[prepared_item.index] = sentences
            candidates.extend(
                _SentenceCandidate(
                    item_index=prepared_item.index,
                    sentence_index=sentence_index,
                    probe=prepared_item.item.model_copy(
                        update={
                            "id": f"{prepared_item.item.id}:sentence:{sentence_index}",
                            "content": sentence,
                            "token_count": None,
                        }
                    ),
                )
                for sentence_index, sentence in enumerate(sentences)
            )
        relevance_started = perf_counter()
        relevance = relevance_scores(
            task,
            [candidate.probe for candidate in candidates],
            provider=self._embedding_provider,
        )
        selected_sentences: dict[int, set[int]] = {}

        def selected_content(item_index: int, extra: int | None = None) -> str:
            selected = selected_sentences.get(item_index, set())
            return " ".join(
                sentence
                for sentence_index, sentence in enumerate(sentences_by_item[item_index])
                if sentence_index in selected or sentence_index == extra
            )

        selection_started = perf_counter()
        timings["relevance"] = (selection_started - relevance_started) * 1000
        for candidate in sorted(
            candidates,
            key=lambda value: (
                -relevance[value.probe.id],
                value.item_index,
                value.sentence_index,
            ),
        ):
            proposed_total = sum(
                tokenizer.count_tokens(
                    selected_content(
                        prepared_item.index,
                        candidate.sentence_index
                        if prepared_item.index == candidate.item_index
                        else None,
                    )
                )
                for prepared_item in prepared
            )
            if proposed_total <= policy.effective_budget:
                selected_sentences.setdefault(candidate.item_index, set()).add(
                    candidate.sentence_index
                )
        selected_items: list[ContextItem] = []
        removed_items: list[ContextItem] = []
        item_traces: list[ItemTrace] = []
        for prepared_item in prepared:
            content = selected_content(prepared_item.index)
            initial_tokens = prepared_item.item.token_count or 0
            if not content:
                removed_items.append(prepared_item.item)
                item_traces.append(
                    ItemTrace(
                        item_id=prepared_item.item.id,
                        initial_token_count=initial_tokens,
                        decision=OptimizationDecision.REMOVED,
                        decision_reason="no_sentence_selected",
                        final_token_count=0,
                        provenance=[prepared_item.item.id],
                    )
                )
                continue
            final_tokens = tokenizer.count_tokens(content)
            selected_item = prepared_item.item.model_copy(
                update={"content": content, "token_count": final_tokens}
            )
            selected_items.append(selected_item)
            compressed = content != prepared_item.item.content
            item_relevance = max(
                (
                    relevance[candidate.probe.id]
                    for candidate in candidates
                    if candidate.item_index == prepared_item.index
                ),
                default=0.0,
            )
            item_traces.append(
                ItemTrace(
                    item_id=prepared_item.item.id,
                    initial_token_count=initial_tokens,
                    relevance_score=item_relevance,
                    decision=(
                        OptimizationDecision.COMPRESSED
                        if compressed
                        else OptimizationDecision.RETAINED
                    ),
                    decision_reason=(
                        "task_relevant_sentences_selected"
                        if compressed
                        else "complete_item_selected"
                    ),
                    final_token_count=final_tokens,
                    final_position=len(selected_items) - 1,
                    compression_strategy="naive_extractive" if compressed else None,
                    provenance=[prepared_item.item.id],
                )
            )
        finished = perf_counter()
        final_tokens = sum(item.token_count or 0 for item in selected_items)
        original_tokens = sum(value.item.token_count or 0 for value in prepared)
        mandatory_tokens = sum(
            value.item.token_count or 0 for value in prepared if value.item.mandatory
        )
        warnings = ["baseline_does_not_enforce_mandatory_retention"] if mandatory_tokens else []
        trace = OptimizationTrace(
            strategy=self.name,
            policy=policy,
            effective_budget=policy.effective_budget,
            mandatory_tokens=mandatory_tokens,
            optional_budget=max(policy.effective_budget - mandatory_tokens, 0),
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            reduction_ratio=(0.0 if original_tokens == 0 else 1 - final_tokens / original_tokens),
            stage_timings_ms={
                **timings,
                "selection": (finished - selection_started) * 1000,
            },
            selected_count=len(selected_items),
            removed_count=len(removed_items),
            compressed_count=sum(
                trace.decision is OptimizationDecision.COMPRESSED for trace in item_traces
            ),
            warnings=warnings,
            items=item_traces,
        )
        return OptimizedContext(
            selected_items=selected_items,
            removed_items=removed_items,
            original_token_count=original_tokens,
            final_token_count=final_tokens,
            budget_allocation=BudgetAllocation(
                effective_budget=policy.effective_budget,
                used_tokens=final_tokens,
                remaining_tokens=policy.effective_budget - final_tokens,
            ),
            trace=trace,
        )
