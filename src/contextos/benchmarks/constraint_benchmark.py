"""Execution, metrics, and artifacts for ContextOS-Bench Constraints."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import mean

from contextos.benchmarks.bundles import capture_environment, write_benchmark_bundle
from contextos.benchmarks.constraint_dataset import PHASE5_BASELINE_SHA
from contextos.benchmarks.constraint_models import (
    ConstraintAggregate,
    ConstraintBenchmarkDataset,
    ConstraintMeasurement,
)
from contextos.benchmarks.models import BenchmarkRun, ContextOSBenchDataset
from contextos.benchmarks.runner import BenchmarkStrategy, run_contextos_bench
from contextos.embeddings import DeterministicEmbeddingProvider
from contextos.models import DependencyRelation
from contextos.tokenization import Tokenizer


def _presence_rate(values: list[str], visible: str) -> float | None:
    if not values:
        return None
    return sum(value in visible for value in values) / len(values)


def evaluate_constraints(
    dataset: ConstraintBenchmarkDataset,
    run: BenchmarkRun,
) -> list[ConstraintMeasurement]:
    """Calculate deterministic preservation metrics from final visible context."""
    cases = {case.id: case for case in dataset.cases}
    results: list[ConstraintMeasurement] = []
    for measurement in run.measurements:
        case = cases[measurement.case_id]
        truth = case.constraints
        selected = set(measurement.selected_item_ids)
        visible = "\n".join(item.content for item in measurement.selected_items)

        active_requires = [
            edge
            for edge in truth.relations
            if edge.relation is DependencyRelation.REQUIRES and edge.source_id in selected
        ]
        dependency_rate = (
            sum(edge.target_id in selected for edge in active_requires) / len(active_requires)
            if active_requires
            else None
        )
        state_checks = [value in visible for value in truth.expected_current_state.values()]
        state_checks.extend(item_id not in selected for item_id in truth.stale_item_ids)
        state_rate = sum(state_checks) / len(state_checks) if state_checks else None
        leakage_rate = (
            sum(set(group) <= selected for group in truth.forbidden_item_combinations)
            / len(truth.forbidden_item_combinations)
            if truth.forbidden_item_combinations
            else None
        )
        exact_rate = _presence_rate(truth.required_exact_values, visible)
        identifier_rate = _presence_rate(truth.required_identifiers, visible)
        citation_rate = _presence_rate(truth.required_citations, visible)
        critical_rate = sum(item_id in selected for item_id in truth.critical_item_ids) / len(
            truth.critical_item_ids
        )

        preservation_rates = [
            critical_rate,
            *(
                value
                for value in (
                    dependency_rate,
                    state_rate,
                    exact_rate,
                    identifier_rate,
                    citation_rate,
                )
                if value is not None
            ),
        ]
        violated = any(value < 1.0 for value in preservation_rates) or (
            leakage_rate is not None and leakage_rate > 0.0
        )
        results.append(
            ConstraintMeasurement(
                benchmark=measurement,
                category=truth.category,
                constraint_violation_rate=float(violated),
                dependency_closure_rate=dependency_rate,
                state_consistency_rate=state_rate,
                contradiction_leakage_rate=leakage_rate,
                exact_value_preservation_rate=exact_rate,
                identifier_preservation_rate=identifier_rate,
                citation_preservation_rate=citation_rate,
            )
        )
    return results


def _optional_mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def aggregate_constraints(
    measurements: list[ConstraintMeasurement],
) -> list[ConstraintAggregate]:
    """Aggregate constraint metrics per strategy without mixing task families."""
    grouped: dict[str, list[ConstraintMeasurement]] = defaultdict(list)
    for measurement in measurements:
        grouped[measurement.benchmark.strategy].append(measurement)
    return [
        ConstraintAggregate(
            strategy=strategy,
            case_count=len(values),
            constraint_violation_rate=mean(value.constraint_violation_rate for value in values),
            dependency_closure_rate=_optional_mean(
                value.dependency_closure_rate for value in values
            ),
            state_consistency_rate=_optional_mean(value.state_consistency_rate for value in values),
            contradiction_leakage_rate=_optional_mean(
                value.contradiction_leakage_rate for value in values
            ),
            exact_value_preservation_rate=_optional_mean(
                value.exact_value_preservation_rate for value in values
            ),
            identifier_preservation_rate=_optional_mean(
                value.identifier_preservation_rate for value in values
            ),
            citation_preservation_rate=_optional_mean(
                value.citation_preservation_rate for value in values
            ),
        )
        for strategy, values in sorted(grouped.items())
    ]


def run_constraint_benchmark(
    dataset: ConstraintBenchmarkDataset,
    *,
    tokenizer: Tokenizer,
    strategies: list[BenchmarkStrategy] | None = None,
    case_limit: int | None = None,
) -> tuple[BenchmarkRun, list[ConstraintMeasurement], list[ConstraintAggregate]]:
    """Run unchanged v0.4 strategies and add constraint-sensitive evaluation."""
    compatible_dataset = ContextOSBenchDataset(
        name=dataset.name,
        generator_version=dataset.generator_version,
        generation_seed=dataset.generation_seed,
        cases=list(dataset.cases),
    )
    run = run_contextos_bench(
        compatible_dataset,
        tokenizer=tokenizer,
        strategies=strategies,
        case_limit=case_limit,
    )
    selected_ids = {measurement.case_id for measurement in run.measurements}
    selected_dataset = dataset.model_copy(
        update={"cases": [case for case in dataset.cases if case.id in selected_ids]}
    )
    measurements = evaluate_constraints(selected_dataset, run)
    return run, measurements, aggregate_constraints(measurements)


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _report(
    run: BenchmarkRun,
    aggregates: list[ConstraintAggregate],
    *,
    current_sha: str,
) -> str:
    base = {value.strategy: value for value in run.aggregates}
    lines = [
        "# ContextOS-Bench Constraints Report",
        "",
        f"- Frozen v0.4 baseline SHA: `{PHASE5_BASELINE_SHA}`",
        f"- Current research SHA: `{current_sha}`",
        f"- Cases: {run.metadata['case_count']}",
        "",
        "| Strategy | Task score | CIR | Reduction | Violation rate | "
        "Dependency closure | State consistency | Contradiction leakage | "
        "Exact values | Identifiers | Citations | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for aggregate in aggregates:
        benchmark = base[aggregate.strategy]
        lines.append(
            f"| {aggregate.strategy} | {benchmark.mean_task_specific_score:.4f} | "
            f"{benchmark.mean_critical_information_recall:.4f} | "
            f"{benchmark.mean_context_reduction:.4f} | "
            f"{aggregate.constraint_violation_rate:.4f} | "
            f"{_format_optional(aggregate.dependency_closure_rate)} | "
            f"{_format_optional(aggregate.state_consistency_rate)} | "
            f"{_format_optional(aggregate.contradiction_leakage_rate)} | "
            f"{_format_optional(aggregate.exact_value_preservation_rate)} | "
            f"{_format_optional(aggregate.identifier_preservation_rate)} | "
            f"{_format_optional(aggregate.citation_preservation_rate)} | "
            f"{benchmark.p95_optimizer_latency_ms:.3f} |"
        )
    lines.extend(
        [
            "",
            "Constraint violation is a binary per-case result. A case violates its "
            "contract when any applicable preservation rate is below one or a "
            "forbidden combination leaks into model-visible context.",
            "",
            "Raw `cases.jsonl` and `predictions.jsonl` are authoritative; this report is derived.",
        ]
    )
    return "\n".join(lines)


def write_constraint_artifact(
    output_directory: Path,
    *,
    dataset: ConstraintBenchmarkDataset,
    run: BenchmarkRun,
    measurements: list[ConstraintMeasurement],
    aggregates: list[ConstraintAggregate],
    profile: str,
) -> Path:
    """Write the immutable seven-file Phase 5 benchmark bundle."""
    selected_ids = {measurement.benchmark.case_id for measurement in measurements}
    cases = [case for case in dataset.cases if case.id in selected_ids]
    embedding = DeterministicEmbeddingProvider()
    environment = capture_environment(
        recorded_at_utc=run.recorded_at_utc,
        embedding_provider="deterministic",
        embedding_model=embedding.model_name,
    )
    current_sha = environment.git_sha
    config = {
        "schema_version": "2.0",
        "benchmark": "contextos-bench-constraints",
        "generator_version": dataset.generator_version,
        "baseline_v040_sha": PHASE5_BASELINE_SHA,
        "current_research_sha": current_sha,
        "dataset_sha256": run.dataset_sha256,
        "profile": profile,
        "strategies": run.strategies,
        "case_count": len(cases),
    }
    metrics = {
        "schema_version": "2.0",
        "run_id": run.run_id,
        "baseline_v040_sha": PHASE5_BASELINE_SHA,
        "current_research_sha": current_sha,
        "benchmark_aggregates": [value.model_dump(mode="json") for value in run.aggregates],
        "constraint_aggregates": [value.model_dump(mode="json") for value in aggregates],
        "paired_comparisons": [value.model_dump(mode="json") for value in run.paired_comparisons],
    }
    rows = [
        {
            "record_type": "constraint_aggregate",
            **value.model_dump(mode="json"),
        }
        for value in aggregates
    ]
    return write_benchmark_bundle(
        output_directory=output_directory,
        recorded_at_utc=run.recorded_at_utc,
        strategy="constraint-comparison",
        profile=profile,
        config=config,
        environment=environment,
        cases=cases,
        predictions=measurements,
        metrics=metrics,
        metric_rows=rows,
        report=_report(run, aggregates, current_sha=current_sha),
    )
