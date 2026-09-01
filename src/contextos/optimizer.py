"""Authoritative integrated ContextOS optimization pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from time import perf_counter
from typing import TypeVar

from contextos.budget import TokenBudgetAllocator, validate_contextual_budget
from contextos.compression import CompressionExecution, CompressionExecutor
from contextos.config import OptimizationPolicy
from contextos.dedup import exact_deduplicate, semantic_deduplicate
from contextos.dedup.base import DeduplicationResult, DuplicateMatch
from contextos.dependency import DependencyGraph
from contextos.embeddings import (
    CachedEmbeddingProvider,
    DeterministicEmbeddingProvider,
    EmbeddingProvider,
)
from contextos.errors import EmbeddingProviderError, MandatoryContextOverflow
from contextos.layout import LayoutStrategy, OriginalOrderLayout, PositionAwareLayout
from contextos.models import ContextEdge, ContextItem, validate_unique_item_ids
from contextos.scoring import ScoreBreakdown
from contextos.scoring.composite import composite_scores
from contextos.scoring.importance import importance_scores
from contextos.scoring.novelty import novelty_scores
from contextos.scoring.recency import recency_scores
from contextos.scoring.relevance import relevance_scores
from contextos.scoring.type_priority import type_priority_scores
from contextos.store import ContextStore
from contextos.tokenization import TiktokenTokenizer, Tokenizer
from contextos.trace import (
    BudgetAllocation,
    ItemTrace,
    OptimizationDecision,
    OptimizationTrace,
    OptimizedContext,
)

T = TypeVar("T")


class ContextOptimizer:
    """Construct final model context as the sole token-budget authority."""

    def __init__(
        self,
        *,
        tokenizer: Tokenizer | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        edges: Sequence[ContextEdge] = (),
        layout: LayoutStrategy | None = None,
        store: ContextStore | None = None,
    ) -> None:
        self._tokenizer = tokenizer or TiktokenTokenizer()
        self._provider = embedding_provider or DeterministicEmbeddingProvider()
        self._edges = tuple(edges)
        self._layout = layout
        self._store = store

    def optimize(
        self,
        task: str,
        items: Sequence[ContextItem],
        policy: OptimizationPolicy,
    ) -> OptimizedContext:
        """Execute every v0.3 stage in the documented order."""
        timings: dict[str, float] = {}
        warnings: list[str] = []

        def stage(name: str, operation: Callable[[], T]) -> T:
            started = perf_counter()
            try:
                return operation()
            finally:
                timings[name] = (perf_counter() - started) * 1000

        stage("validate_policy", policy.validate_static)
        original_positions = {item.id: index for index, item in enumerate(items)}

        def tokenize() -> list[ContextItem]:
            validate_unique_item_ids(items)
            tokenized: list[ContextItem] = []
            for item in items:
                copied = item.model_copy(deep=True)
                copied.token_count = self._tokenizer.count_tokens(copied.content)
                tokenized.append(copied)
            return tokenized

        tokenized = stage("tokenize", tokenize)
        original_tokens = sum(item.token_count or 0 for item in tokenized)

        def reserve_mandatory() -> int:
            mandatory_tokens = sum(item.token_count or 0 for item in tokenized if item.mandatory)
            if mandatory_tokens > policy.effective_budget:
                raise MandatoryContextOverflow(
                    mandatory_tokens=mandatory_tokens,
                    effective_budget=policy.effective_budget,
                )
            return mandatory_tokens

        stage("reserve_mandatory", reserve_mandatory)
        exact = stage("exact_dedup", lambda: exact_deduplicate(tokenized))
        warnings.extend(exact.warnings)

        provider: EmbeddingProvider = CachedEmbeddingProvider(self._provider)

        def run_semantic() -> DeduplicationResult:
            return semantic_deduplicate(
                exact.items,
                provider=provider,
                threshold=policy.semantic_dedup_threshold,
            )

        if policy.semantic_dedup_enabled:
            try:
                semantic = stage("semantic_dedup", run_semantic)
            except EmbeddingProviderError:
                warnings.append("embedding_provider_unavailable:deterministic_fallback")
                provider = CachedEmbeddingProvider(DeterministicEmbeddingProvider())
                semantic = stage("semantic_dedup", run_semantic)
        else:
            semantic = stage(
                "semantic_dedup",
                lambda: DeduplicationResult(items=exact.items),
            )
        warnings.extend(semantic.warnings)
        survivors = semantic.items
        contextual = stage(
            "contextual_budget_validation",
            lambda: validate_contextual_budget(survivors, policy=policy),
        )

        relevance = stage(
            "relevance",
            lambda: (
                relevance_scores(task, survivors, provider=provider)
                if policy.semantic_relevance_enabled
                else {item.id: 0.0 for item in survivors}
            ),
        )
        importance = stage("importance", lambda: importance_scores(survivors))
        recency = stage(
            "recency",
            lambda: (
                recency_scores(
                    survivors,
                    half_life_seconds=policy.recency_half_life_seconds,
                )
                if policy.weight_recency > 0
                else {item.id: 0.0 for item in survivors}
            ),
        )
        novelty = stage("novelty", lambda: novelty_scores(survivors, provider=provider))
        for item in survivors:
            if item.mandatory:
                novelty[item.id] = 1.0

        def dependency_stage() -> dict[str, float]:
            if policy.weight_dependency == 0:
                return {item.id: 0.0 for item in survivors}
            DependencyGraph([item.id for item in tokenized], self._edges)
            survivor_ids = {item.id for item in survivors}
            survivor_edges = [
                edge
                for edge in self._edges
                if edge.source_id in survivor_ids and edge.target_id in survivor_ids
            ]
            graph = DependencyGraph(list(survivor_ids), survivor_edges)
            return graph.propagate_scores(
                {item.id: max(relevance[item.id], importance[item.id]) for item in survivors},
                max_depth=policy.dependency_max_depth,
            )

        dependency = stage("dependencies", dependency_stage)
        type_priority = type_priority_scores(survivors, priorities=policy.type_priorities)
        scores = stage(
            "composite_utility",
            lambda: composite_scores(
                survivors,
                policy=policy,
                relevance=relevance,
                importance=importance,
                recency=recency,
                novelty=novelty,
                dependency=dependency,
                type_priority=type_priority,
            ),
        )
        plan = stage(
            "allocation_plan",
            lambda: TokenBudgetAllocator().allocate(survivors, scores=scores, policy=policy),
        )
        compression = stage(
            "compression",
            lambda: CompressionExecutor(self._tokenizer).execute(
                plan,
                survivors,
                task=task,
                policy=policy,
            ),
        )

        def final_selection() -> list[ContextItem]:
            survivor_by_id = {item.id: item for item in survivors}
            selected = [item.model_copy(deep=True) for item in survivors if item.mandatory]
            selected.extend(
                survivor_by_id[selection.item_id].model_copy(deep=True)
                for selection in plan.direct_selected
            )
            for item_id, result in compression.successful_results.items():
                source = survivor_by_id[item_id].model_copy(deep=True)
                source.content = result.content or source.content
                source.token_count = result.compressed_tokens
                source.metadata = {
                    **source.metadata,
                    "compression_strategy": result.strategy,
                    "compression_provenance": list(result.provenance),
                    "compression_lossy": result.lossy,
                }
                selected.append(source)
            return selected

        selected = stage("final_selection", final_selection)
        strategy = self._layout or (
            PositionAwareLayout() if policy.position_aware_layout else OriginalOrderLayout()
        )
        laid_out = stage(
            "layout",
            lambda: strategy.arrange(
                selected,
                scores=scores,
                original_positions=original_positions,
            ),
        )

        def validate_invariants() -> int:
            validate_unique_item_ids(laid_out)
            selected_ids = {item.id for item in laid_out}
            missing_mandatory = sorted(
                item.id for item in survivors if item.mandatory and item.id not in selected_ids
            )
            if missing_mandatory:
                raise AssertionError("mandatory context was removed")
            final_tokens = sum(item.token_count or 0 for item in laid_out)
            if final_tokens > policy.effective_budget:
                raise AssertionError("integrated optimizer exceeded the effective budget")
            return final_tokens

        final_tokens = stage("invariant_validation", validate_invariants)
        exact_matches = exact.matches_by_item_id
        semantic_matches = semantic.matches_by_item_id
        traces = stage(
            "trace",
            lambda: self._build_item_traces(
                tokenized,
                laid_out,
                scores,
                plan.rejection_reasons,
                compression,
                exact_matches,
                semantic_matches,
            ),
        )

        def persist() -> None:
            if self._store is None:
                return
            for item in tokenized:
                self._store.save_item(item)
            for edge in self._edges:
                self._store.save_edge(edge)

        stage("lifecycle_persistence", persist)
        selected_ids = {item.id for item in laid_out}
        removed = [item for item in tokenized if item.id not in selected_ids]
        reduction = (
            0.0 if original_tokens == 0 else (original_tokens - final_tokens) / original_tokens
        )
        trace = OptimizationTrace(
            strategy="contextos",
            policy=policy,
            effective_budget=policy.effective_budget,
            mandatory_tokens=contextual.mandatory_tokens,
            optional_budget=contextual.optional_budget,
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            reduction_ratio=max(0.0, min(reduction, 1.0)),
            stage_timings_ms=timings,
            selected_count=len(laid_out),
            removed_count=len(removed),
            compressed_count=len(compression.successful_results),
            warnings=sorted(set(warnings)),
            items=traces,
        )
        return OptimizedContext(
            selected_items=laid_out,
            removed_items=removed,
            original_token_count=original_tokens,
            final_token_count=final_tokens,
            budget_allocation=BudgetAllocation(
                effective_budget=policy.effective_budget,
                used_tokens=final_tokens,
                remaining_tokens=policy.effective_budget - final_tokens,
            ),
            trace=trace,
            metadata={"layout": type(strategy).__name__},
        )

    @staticmethod
    def _build_item_traces(
        original: Sequence[ContextItem],
        selected: Sequence[ContextItem],
        scores: Mapping[str, ScoreBreakdown],
        rejection_reasons: Mapping[str, str],
        compression: CompressionExecution,
        exact_matches: Mapping[str, DuplicateMatch],
        semantic_matches: Mapping[str, DuplicateMatch],
    ) -> list[ItemTrace]:
        selected_by_id = {item.id: item for item in selected}
        positions = {item.id: index for index, item in enumerate(selected)}
        attempts = {attempt.item_id: attempt for attempt in compression.attempts}
        traces: list[ItemTrace] = []
        for item in original:
            exact = exact_matches.get(item.id)
            semantic = semantic_matches.get(item.id)
            score = scores.get(item.id)
            final = selected_by_id.get(item.id)
            result = compression.successful_results.get(item.id)
            if exact is not None:
                decision = OptimizationDecision.REMOVED
                reason = exact.reason
            elif semantic is not None:
                decision = OptimizationDecision.REMOVED
                reason = semantic.reason
            elif result is not None:
                decision = OptimizationDecision.COMPRESSED
                reason = "compressed_to_fit_budget"
            elif final is not None:
                decision = OptimizationDecision.RETAINED
                reason = "mandatory" if item.mandatory else "allocated_without_compression"
            else:
                decision = OptimizationDecision.REMOVED
                attempt = attempts.get(item.id)
                reason = (
                    attempt.reason
                    if attempt is not None and attempt.reason is not None
                    else rejection_reasons.get(item.id, "not_selected")
                )
            traces.append(
                ItemTrace(
                    item_id=item.id,
                    initial_token_count=item.token_count or 0,
                    exact_duplicate_of=exact.duplicate_of if exact else None,
                    semantic_duplicate_of=semantic.duplicate_of if semantic else None,
                    semantic_similarity=semantic.similarity if semantic else None,
                    relevance_score=score.relevance if score else None,
                    importance_score=score.importance if score else None,
                    recency_score=score.recency if score else None,
                    novelty_score=score.novelty if score else None,
                    dependency_score=score.dependency if score else None,
                    type_priority=score.type_priority if score else None,
                    composite_utility=score.composite_utility if score else None,
                    value_density=(
                        score.composite_utility / max(item.token_count or 0, 1) if score else None
                    ),
                    decision=decision,
                    decision_reason=reason,
                    final_token_count=final.token_count or 0 if final else 0,
                    final_position=positions.get(item.id),
                    compression_strategy=result.strategy if result else None,
                    provenance=list(result.provenance) if result else [],
                )
            )
        return traces
