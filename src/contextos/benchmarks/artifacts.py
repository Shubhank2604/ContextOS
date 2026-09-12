"""ContextOS-Bench dataset loading and immutable bundle adaptation."""

from __future__ import annotations

from pathlib import Path

from contextos.benchmarks.bundles import capture_environment, write_benchmark_bundle
from contextos.benchmarks.metrics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    MIN_BOOTSTRAP_SAMPLE_SIZE,
)
from contextos.benchmarks.models import BenchmarkRun, ContextOSBenchDataset
from contextos.embeddings import DeterministicEmbeddingProvider


def _display_optional(value: int | None) -> int | str:
    return "n/a" if value is None else value


def load_dataset(path: Path) -> ContextOSBenchDataset:
    """Load and validate a versioned ContextOS-Bench dataset."""
    return ContextOSBenchDataset.model_validate_json(path.read_text(encoding="utf-8"))


def _report(run: BenchmarkRun, *, profile: str, strategy_label: str) -> str:
    lines = [
        "# ContextOS-Bench Report",
        "",
        f"- Run ID: `{run.run_id}`",
        f"- Profile: `{profile}`",
        f"- Strategy group: `{strategy_label}`",
        f"- Cases: {run.metadata['case_count']}",
        "",
        "| Strategy | Task score | CIR | Mean input tokens | p95 optimizer ms |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        "| "
        f"{aggregate.strategy} | {aggregate.mean_task_specific_score:.4f} | "
        f"{aggregate.mean_critical_information_recall:.4f} | "
        f"{aggregate.mean_input_tokens:.2f} | "
        f"{aggregate.p95_optimizer_latency_ms:.3f} |"
        for aggregate in run.aggregates
    )
    lines.extend(
        [
            "",
            "## Optimizer performance",
            "",
            f"Observed process peak resident memory: "
            f"{_display_optional(run.peak_process_memory_bytes)} bytes. "
            "It is process-wide and must not be attributed to an individual strategy.",
            "",
            "| Strategy | Total ms | p50 ms | p95 ms | Mean embedding ms | Mean compression ms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        "| "
        f"{aggregate.strategy} | {aggregate.total_optimizer_wall_time_ms:.3f} | "
        f"{aggregate.p50_optimizer_latency_ms:.3f} | "
        f"{aggregate.p95_optimizer_latency_ms:.3f} | "
        f"{aggregate.mean_embedding_time_ms:.3f} | "
        f"{aggregate.mean_compression_time_ms:.3f} |"
        for aggregate in run.aggregates
    )
    if run.paired_comparisons:
        lines.extend(
            [
                "",
                "## Paired deltas",
                "",
                "Candidate minus reference; intervals are omitted below 20 paired cases.",
                "",
                "| Reference | Candidate | Metric | Cases | Mean delta | 95% CI |",
                "|---|---|---|---:|---:|---|",
            ]
        )
        lines.extend(
            "| "
            f"{value.reference_strategy} | {value.candidate_strategy} | {value.metric} | "
            f"{value.case_count} | {value.mean_delta:.4f} | "
            + (
                f"[{value.delta_ci95.low:.4f}, {value.delta_ci95.high:.4f}] |"
                if value.delta_ci95 is not None
                else "not reported |"
            )
            for value in run.paired_comparisons
        )
    lines.extend(
        [
            "",
            "Raw `cases.jsonl` and `predictions.jsonl` are authoritative; this report is derived.",
        ]
    )
    return "\n".join(lines)


def write_run_artifact(
    run: BenchmarkRun,
    output_directory: Path,
    *,
    dataset: ContextOSBenchDataset,
    profile: str,
    strategy_label: str,
) -> Path:
    """Write a complete immutable ContextOS-Bench artifact bundle."""
    case_ids = {measurement.case_id for measurement in run.measurements}
    cases = [case for case in dataset.cases if case.id in case_ids]
    if len(cases) != run.metadata["case_count"]:
        raise ValueError("benchmark run cases do not match the supplied dataset")
    embedding = DeterministicEmbeddingProvider()
    environment = capture_environment(
        recorded_at_utc=run.recorded_at_utc,
        embedding_provider="deterministic",
        embedding_model=embedding.model_name,
    )
    config = {
        "schema_version": "1.0",
        "benchmark": "contextos-bench",
        "run_id": run.run_id,
        "dataset_name": run.dataset_name,
        "dataset_sha256": run.dataset_sha256,
        "profile": profile,
        "strategy_group": strategy_label,
        "strategies": run.strategies,
        "metadata": run.metadata,
        "statistics": {
            "method": "percentile bootstrap; paired for strategy deltas",
            "confidence_level": 0.95,
            "resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
            "minimum_sample_size": MIN_BOOTSTRAP_SAMPLE_SIZE,
        },
        "performance_measurement": {
            "wall_clock": "time.perf_counter",
            "memory": "process-lifetime peak resident set from native OS counters",
            "memory_is_process_rss": True,
        },
    }
    metrics = {
        "schema_version": run.schema_version,
        "run_id": run.run_id,
        "aggregates": [aggregate.model_dump(mode="json") for aggregate in run.aggregates],
        "paired_comparisons": [
            comparison.model_dump(mode="json") for comparison in run.paired_comparisons
        ],
        "performance": {
            "peak_process_memory_bytes": run.peak_process_memory_bytes,
        },
    }
    metric_rows = [
        {"record_type": "aggregate", **aggregate.model_dump(mode="json")}
        for aggregate in run.aggregates
    ]
    metric_rows.extend(
        {"record_type": "paired_comparison", **comparison.model_dump(mode="json")}
        for comparison in run.paired_comparisons
    )
    return write_benchmark_bundle(
        output_directory=output_directory,
        recorded_at_utc=run.recorded_at_utc,
        strategy=strategy_label,
        profile=profile,
        config=config,
        environment=environment,
        cases=cases,
        predictions=run.measurements,
        metrics=metrics,
        metric_rows=metric_rows,
        report=_report(run, profile=profile, strategy_label=strategy_label),
    )
