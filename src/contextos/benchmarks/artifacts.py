"""ContextOS-Bench dataset loading and immutable bundle adaptation."""

from __future__ import annotations

from pathlib import Path

from contextos.benchmarks.bundles import capture_environment, write_benchmark_bundle
from contextos.benchmarks.models import BenchmarkRun, ContextOSBenchDataset
from contextos.embeddings import DeterministicEmbeddingProvider


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
    }
    metrics = {
        "schema_version": "1.0",
        "run_id": run.run_id,
        "aggregates": [aggregate.model_dump(mode="json") for aggregate in run.aggregates],
    }
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
        metric_rows=[aggregate.model_dump(mode="json") for aggregate in run.aggregates],
        report=_report(run, profile=profile, strategy_label=strategy_label),
    )
