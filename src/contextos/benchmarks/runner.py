"""Shared baseline and ContextOS runner for ContextOS-Bench."""

from __future__ import annotations

import hashlib
import platform
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from time import perf_counter
from typing import Protocol

from contextos import __version__
from contextos.baselines import (
    BaselineStrategy,
    FullContextBaseline,
    LastNTokensBaseline,
    NaiveExtractiveBaseline,
    RelevanceOnlyBaseline,
    SlidingWindowBaseline,
)
from contextos.benchmarks.metrics import (
    aggregate_measurements,
    failed_measurement,
    measurement_from_result,
)
from contextos.benchmarks.models import (
    BenchmarkMeasurement,
    BenchmarkRun,
    ContextOSBenchCase,
    ContextOSBenchDataset,
)
from contextos.errors import ContextBudgetOverflow
from contextos.optimizer import ContextOptimizer
from contextos.tokenization import Tokenizer
from contextos.trace import OptimizedContext


class BenchmarkStrategy(Protocol):
    """Uniform runner interface for baselines and ContextOS."""

    @property
    def name(self) -> str:
        """Return a stable strategy identifier."""
        ...

    def optimize(self, case: ContextOSBenchCase, tokenizer: Tokenizer) -> OptimizedContext:
        """Optimize one benchmark case."""
        ...


class BaselineBenchmarkStrategy:
    """Adapt an existing baseline to the benchmark runner interface."""

    def __init__(
        self,
        baseline: BaselineStrategy,
        *,
        full_context_reference: bool = False,
    ) -> None:
        self._baseline = baseline
        self._full_context_reference = full_context_reference

    @property
    def name(self) -> str:
        return self._baseline.name

    def optimize(self, case: ContextOSBenchCase, tokenizer: Tokenizer) -> OptimizedContext:
        policy = case.policy
        if self._full_context_reference:
            original_tokens = sum(
                tokenizer.count_tokens(item.content) for item in case.context_items
            )
            policy = case.policy.model_copy(
                update={
                    "max_input_tokens": original_tokens + case.policy.reserve_output_tokens,
                }
            )
        return self._baseline.optimize(
            task=case.task,
            items=case.context_items,
            policy=policy,
            tokenizer=tokenizer,
        )


class ContextOSBenchmarkStrategy:
    """Adapt the integrated runtime without changing its public budget authority."""

    def __init__(
        self,
        *,
        name: str = "contextos",
        policy_overrides: Mapping[str, object] | None = None,
    ) -> None:
        self._name = name
        self._policy_overrides = dict(policy_overrides or {})

    @property
    def name(self) -> str:
        return self._name

    @property
    def policy_overrides(self) -> dict[str, object]:
        """Return a copy of the single-variant policy changes for auditing."""
        return dict(self._policy_overrides)

    def optimize(self, case: ContextOSBenchCase, tokenizer: Tokenizer) -> OptimizedContext:
        policy = case.policy.model_copy(update=self._policy_overrides)
        return ContextOptimizer(tokenizer=tokenizer, edges=case.edges).optimize(
            case.task,
            case.context_items,
            policy,
        )


def default_benchmark_strategies() -> list[BenchmarkStrategy]:
    """Return the complete required Phase 4D comparison in stable order."""
    return [
        BaselineBenchmarkStrategy(
            FullContextBaseline(),
            full_context_reference=True,
        ),
        BaselineBenchmarkStrategy(LastNTokensBaseline()),
        BaselineBenchmarkStrategy(SlidingWindowBaseline(window_seconds=7 * 86_400)),
        BaselineBenchmarkStrategy(RelevanceOnlyBaseline()),
        BaselineBenchmarkStrategy(NaiveExtractiveBaseline()),
        ContextOSBenchmarkStrategy(),
    ]


def default_ablation_strategies() -> list[ContextOSBenchmarkStrategy]:
    """Return the six Phase 4E single-component ContextOS variants."""
    return [
        ContextOSBenchmarkStrategy(name="contextos_full"),
        ContextOSBenchmarkStrategy(
            name="contextos_without_semantic_deduplication",
            policy_overrides={"semantic_dedup_enabled": False},
        ),
        ContextOSBenchmarkStrategy(
            name="contextos_without_recency",
            policy_overrides={"weight_recency": 0.0},
        ),
        ContextOSBenchmarkStrategy(
            name="contextos_without_dependency_score",
            policy_overrides={"weight_dependency": 0.0},
        ),
        ContextOSBenchmarkStrategy(
            name="contextos_without_compression",
            policy_overrides={"compression_enabled": False},
        ),
        ContextOSBenchmarkStrategy(
            name="contextos_without_position_aware_layout",
            policy_overrides={"position_aware_layout": False},
        ),
    ]


def ablation_effects(run: BenchmarkRun) -> dict[str, dict[str, float]]:
    """Calculate required Phase 4E metric deltas against full ContextOS."""
    aggregates = {aggregate.strategy: aggregate for aggregate in run.aggregates}
    try:
        reference = aggregates["contextos_full"]
    except KeyError:
        raise ValueError("ablation run requires a contextos_full reference") from None
    return {
        strategy: {
            "task_score_delta": aggregate.mean_task_specific_score
            - reference.mean_task_specific_score,
            "cir_delta": aggregate.mean_critical_information_recall
            - reference.mean_critical_information_recall,
            "input_token_delta": aggregate.mean_input_tokens - reference.mean_input_tokens,
            "p95_optimizer_latency_ms_delta": aggregate.p95_optimizer_latency_ms
            - reference.p95_optimizer_latency_ms,
        }
        for strategy, aggregate in aggregates.items()
        if strategy != "contextos_full"
    }


def _run_case(
    case: ContextOSBenchCase,
    strategy: BenchmarkStrategy,
    tokenizer: Tokenizer,
) -> BenchmarkMeasurement:
    started = perf_counter()
    try:
        result = strategy.optimize(case, tokenizer)
    except ContextBudgetOverflow as exc:
        elapsed = (perf_counter() - started) * 1000
        return failed_measurement(
            case,
            strategy=strategy.name,
            status="overflow",
            optimizer_wall_time_ms=elapsed,
            warning=str(exc),
        )
    elapsed = (perf_counter() - started) * 1000
    return measurement_from_result(
        case,
        result,
        strategy=strategy.name,
        optimizer_wall_time_ms=elapsed,
    )


def run_contextos_bench(
    dataset: ContextOSBenchDataset,
    *,
    tokenizer: Tokenizer,
    strategies: Sequence[BenchmarkStrategy] | None = None,
    case_limit: int | None = None,
) -> BenchmarkRun:
    """Run selected cases and retain exact raw per-case measurements."""
    if case_limit is not None and case_limit <= 0:
        raise ValueError("case limit must be positive")
    selected_cases = dataset.cases[:case_limit] if case_limit is not None else dataset.cases
    selected_strategies = list(strategies or default_benchmark_strategies())
    if not selected_strategies:
        raise ValueError("at least one benchmark strategy is required")
    strategy_names = [strategy.name for strategy in selected_strategies]
    if len(strategy_names) != len(set(strategy_names)):
        raise ValueError("benchmark strategy names must be unique")
    raw_measurements = [
        _run_case(case, strategy, tokenizer)
        for case in selected_cases
        for strategy in selected_strategies
    ]
    full_scores = {
        measurement.case_id: measurement.task_specific_score
        for measurement in raw_measurements
        if measurement.strategy == "full_context" and measurement.status == "ok"
    }
    measurements: list[BenchmarkMeasurement] = []
    for measurement in raw_measurements:
        full_score = full_scores.get(measurement.case_id)
        quality_retention = (
            measurement.task_specific_score / full_score
            if measurement.status == "ok" and full_score is not None and full_score > 0
            else None
        )
        measurements.append(measurement.model_copy(update={"quality_retention": quality_retention}))
    dataset_payload = dataset.model_dump_json()
    dataset_sha = hashlib.sha256(dataset_payload.encode("utf-8")).hexdigest()
    recorded_at = datetime.now(UTC).isoformat()
    identity_payload = "|".join(
        [
            dataset_sha,
            recorded_at,
            *strategy_names,
            *(measurement.model_dump_json() for measurement in measurements),
        ]
    )
    run_id = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()[:20]
    return BenchmarkRun(
        run_id=run_id,
        recorded_at_utc=datetime.fromisoformat(recorded_at),
        dataset_name=dataset.name,
        dataset_sha256=dataset_sha,
        package_version=__version__,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        strategies=strategy_names,
        measurements=measurements,
        aggregates=aggregate_measurements(
            measurements,
            bootstrap_seed=dataset.generation_seed,
        ),
        metadata={
            "case_count": len(selected_cases),
            "base_case_count": sum(case in dataset.base_cases for case in selected_cases),
            "generator_version": dataset.generator_version,
            "strategy_configurations": {
                strategy.name: strategy.policy_overrides
                for strategy in selected_strategies
                if isinstance(strategy, ContextOSBenchmarkStrategy)
            },
        },
    )
