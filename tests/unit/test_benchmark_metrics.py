"""Deterministic metric, aggregation, and artifact tests."""

from pathlib import Path

import pytest

from contextos.benchmarks.artifacts import load_dataset, write_run_artifact
from contextos.benchmarks.metrics import bootstrap_mean_ci, percentile
from contextos.benchmarks.runner import run_contextos_bench
from contextos.tokenization import TiktokenTokenizer


def test_percentile_interpolates_and_validates_inputs() -> None:
    assert percentile([0.0, 10.0], 0.5) == 5.0
    assert percentile([1.0, 2.0, 3.0], 0.95) == pytest.approx(2.9)
    with pytest.raises(ValueError, match="at least one"):
        percentile([], 0.5)
    with pytest.raises(ValueError, match="between"):
        percentile([1.0], 1.1)


def test_bootstrap_interval_is_reproducible() -> None:
    values = [0.0, 0.5, 1.0, 1.0]
    first = bootstrap_mean_ci(values, seed=17, resamples=100)
    second = bootstrap_mean_ci(values, seed=17, resamples=100)

    assert first == second
    assert 0.0 <= first.low <= first.high <= 1.0


def test_runner_scores_full_reference_and_contextos_on_same_case() -> None:
    dataset = load_dataset(Path("benchmarks/datasets/contextos_bench.json"))
    run = run_contextos_bench(
        dataset,
        tokenizer=TiktokenTokenizer(),
        case_limit=1,
    )
    measurements = {measurement.strategy: measurement for measurement in run.measurements}

    assert set(measurements) == {
        "contextos",
        "full_context",
        "last_n",
        "naive_extractive",
        "relevance_only",
        "sliding_window",
    }
    assert measurements["full_context"].status == "ok"
    assert measurements["full_context"].task_specific_score == 1.0
    assert measurements["full_context"].quality_retention == 1.0
    assert measurements["full_context"].context_reduction == 0.0
    assert measurements["contextos"].quality_retention is not None
    assert measurements["contextos"].input_tokens <= measurements["contextos"].effective_budget
    assert measurements["contextos"].selected_items
    assert measurements["contextos"].decision_reasons


def test_run_artifacts_are_content_addressed_and_immutable(tmp_path: Path) -> None:
    dataset = load_dataset(Path("benchmarks/datasets/contextos_bench.json"))
    run = run_contextos_bench(
        dataset,
        tokenizer=TiktokenTokenizer(),
        case_limit=1,
    )
    path = write_run_artifact(run, tmp_path)

    assert path == write_run_artifact(run, tmp_path)
    assert path.name == f"contextos-bench-{run.run_id}.json"
    conflicting = run.model_copy(update={"metadata": {"different": True}})
    with pytest.raises(ValueError, match="collision"):
        write_run_artifact(conflicting, tmp_path)
