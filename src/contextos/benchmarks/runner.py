"""Shared baseline and ContextOS runner for ContextOS-Bench."""

from __future__ import annotations

import hashlib
import platform
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from time import perf_counter
from typing import Protocol

from contextos import __version__
from contextos.baselines import (
    BaselineStrategy,
    FullContextBaseline,
    LastNTokensBaseline,
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

    @property
    def name(self) -> str:
        return "contextos"

    def optimize(self, case: ContextOSBenchCase, tokenizer: Tokenizer) -> OptimizedContext:
        return ContextOptimizer(tokenizer=tokenizer, edges=case.edges).optimize(
            case.task,
            case.context_items,
            case.policy,
        )


def default_benchmark_strategies() -> list[BenchmarkStrategy]:
    """Return the Phase 4A comparison strategies in stable order."""
    return [
        BaselineBenchmarkStrategy(
            FullContextBaseline(),
            full_context_reference=True,
        ),
        BaselineBenchmarkStrategy(LastNTokensBaseline()),
        BaselineBenchmarkStrategy(SlidingWindowBaseline(window_seconds=7 * 86_400)),
        ContextOSBenchmarkStrategy(),
    ]


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
        },
    )
