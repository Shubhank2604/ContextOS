"""Deterministic ContextOS-Bench per-case and aggregate metrics."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence
from statistics import mean

from contextos.benchmarks.models import (
    BenchmarkAggregate,
    BenchmarkMeasurement,
    ConfidenceInterval,
    ContextOSBenchCase,
)
from contextos.trace import OptimizedContext


def retained_fact_labels(
    case: ContextOSBenchCase,
    result: OptimizedContext,
) -> set[str]:
    """Return labels whose exact values survive in their selected source item."""
    selected_contents = [item.content for item in result.selected_items]
    return {
        fact.label
        for fact in case.answer_key.required_facts
        if any(fact.value in content for content in selected_contents)
    }


def retained_required_item_ids(
    case: ContextOSBenchCase,
    result: OptimizedContext,
) -> set[str]:
    """Count an item only when every annotated fact for it survives."""
    retained_labels = retained_fact_labels(case, result)
    return {
        item_id
        for item_id in case.required_item_ids
        if all(
            fact.label in retained_labels
            for fact in case.answer_key.required_facts
            if fact.item_id == item_id
        )
    }


def measurement_from_result(
    case: ContextOSBenchCase,
    result: OptimizedContext,
    *,
    strategy: str,
    optimizer_wall_time_ms: float,
) -> BenchmarkMeasurement:
    """Create all required raw metrics from one successful strategy result."""
    retained_facts = retained_fact_labels(case, result)
    retained_items = retained_required_item_ids(case, result)
    fact_total = len(case.answer_key.required_facts)
    item_total = len(case.required_item_ids)
    original_tokens = result.original_token_count
    input_tokens = result.final_token_count
    reduction = 0.0 if original_tokens == 0 else 1 - input_tokens / original_tokens
    ratio = None if input_tokens == 0 else original_tokens / input_tokens
    timings = result.trace.stage_timings_ms
    embedding_time = sum(
        timings.get(stage, 0.0) for stage in ("semantic_dedup", "relevance", "novelty")
    )
    return BenchmarkMeasurement(
        case_id=case.id,
        family=case.family,
        strategy=strategy,
        status="ok",
        task_metric=case.task_metric,
        task_specific_score=len(retained_facts) / fact_total,
        critical_information_recall=len(retained_items) / item_total,
        original_tokens=original_tokens,
        input_tokens=input_tokens,
        effective_budget=result.budget_allocation.effective_budget,
        context_reduction=max(0.0, min(reduction, 1.0)),
        compression_ratio=ratio,
        selected_item_ids=[item.id for item in result.selected_items],
        selected_items=result.selected_items,
        required_items_retained=len(retained_items),
        required_items_total=item_total,
        required_facts_retained=len(retained_facts),
        required_facts_total=fact_total,
        compressed_count=result.trace.compressed_count,
        warnings=result.trace.warnings,
        optimizer_wall_time_ms=optimizer_wall_time_ms,
        embedding_time_ms=embedding_time,
        compression_time_ms=timings.get("compression", 0.0),
        stage_timings_ms=timings,
        decision_reasons={trace.item_id: trace.decision_reason for trace in result.trace.items},
    )


def failed_measurement(
    case: ContextOSBenchCase,
    *,
    strategy: str,
    status: str,
    optimizer_wall_time_ms: float,
    warning: str,
) -> BenchmarkMeasurement:
    """Record an expected strategy failure without inventing quality measurements."""
    return BenchmarkMeasurement(
        case_id=case.id,
        family=case.family,
        strategy=strategy,
        status=status,
        task_metric=case.task_metric,
        task_specific_score=0.0,
        critical_information_recall=0.0,
        original_tokens=0,
        input_tokens=0,
        effective_budget=case.policy.effective_budget,
        context_reduction=0.0,
        selected_item_ids=[],
        selected_items=[],
        required_items_retained=0,
        required_items_total=len(case.required_item_ids),
        required_facts_retained=0,
        required_facts_total=len(case.answer_key.required_facts),
        compressed_count=0,
        warnings=[warning],
        optimizer_wall_time_ms=optimizer_wall_time_ms,
        embedding_time_ms=0.0,
        compression_time_ms=0.0,
    )


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for non-empty values."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int = 1_000,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """Calculate a reproducible percentile-bootstrap interval for a mean."""
    if len(values) < 2:
        raise ValueError("bootstrap confidence interval requires at least two values")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must be between zero and one")
    generator = random.Random(seed)
    sample_size = len(values)
    means = [
        mean(values[generator.randrange(sample_size)] for _ in range(sample_size))
        for _ in range(resamples)
    ]
    tail = (1 - confidence_level) / 2
    return ConfidenceInterval(
        low=percentile(means, tail),
        high=percentile(means, 1 - tail),
        confidence_level=confidence_level,
        resamples=resamples,
        seed=seed,
    )


def aggregate_measurements(
    measurements: Sequence[BenchmarkMeasurement],
    *,
    bootstrap_seed: int,
) -> list[BenchmarkAggregate]:
    """Aggregate raw measurements independently for every strategy."""
    aggregates: list[BenchmarkAggregate] = []
    for strategy in sorted({measurement.strategy for measurement in measurements}):
        strategy_measurements = [
            measurement for measurement in measurements if measurement.strategy == strategy
        ]
        successful = [
            measurement for measurement in strategy_measurements if measurement.status == "ok"
        ]
        task_scores = [measurement.task_specific_score for measurement in successful]
        quality_retentions = [
            measurement.quality_retention
            for measurement in successful
            if measurement.quality_retention is not None
        ]
        cir_scores = [measurement.critical_information_recall for measurement in successful]
        ratios = [
            measurement.compression_ratio
            for measurement in successful
            if measurement.compression_ratio is not None
        ]
        latencies = [measurement.optimizer_wall_time_ms for measurement in successful]
        strategy_seed = bootstrap_seed + int.from_bytes(
            hashlib.sha256(strategy.encode("utf-8")).digest()[:4], "big"
        )
        enough_for_ci = len(successful) >= 20
        aggregates.append(
            BenchmarkAggregate(
                strategy=strategy,
                case_count=len(strategy_measurements),
                successful_case_count=len(successful),
                mean_task_specific_score=mean(task_scores) if task_scores else 0.0,
                mean_quality_retention=(mean(quality_retentions) if quality_retentions else None),
                mean_critical_information_recall=mean(cir_scores) if cir_scores else 0.0,
                mean_input_tokens=mean(measurement.input_tokens for measurement in successful)
                if successful
                else 0.0,
                mean_context_reduction=mean(
                    measurement.context_reduction for measurement in successful
                )
                if successful
                else 0.0,
                mean_compression_ratio=mean(ratios) if ratios else None,
                p50_optimizer_latency_ms=percentile(latencies, 0.5) if latencies else 0.0,
                p95_optimizer_latency_ms=percentile(latencies, 0.95) if latencies else 0.0,
                task_score_ci95=(
                    bootstrap_mean_ci(task_scores, seed=strategy_seed) if enough_for_ci else None
                ),
                cir_ci95=(
                    bootstrap_mean_ci(cir_scores, seed=strategy_seed + 1) if enough_for_ci else None
                ),
            )
        )
    return aggregates
