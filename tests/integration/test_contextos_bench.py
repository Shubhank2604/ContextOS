"""End-to-end Phase 4A benchmark execution."""

from pathlib import Path

from contextos.benchmarks.artifacts import load_dataset
from contextos.benchmarks.runner import run_contextos_bench
from contextos.tokenization import TiktokenTokenizer


def test_all_fifty_base_cases_produce_raw_results_and_confidence_intervals() -> None:
    dataset = load_dataset(Path("benchmarks/datasets/contextos_bench.json"))
    run = run_contextos_bench(dataset, tokenizer=TiktokenTokenizer())

    assert run.metadata["case_count"] == 50
    assert run.metadata["base_case_count"] == 50
    assert len(run.measurements) == 200
    assert len(run.aggregates) == 4
    assert all(aggregate.case_count == 50 for aggregate in run.aggregates)
    assert all(aggregate.task_score_ci95 is not None for aggregate in run.aggregates)
    assert all(aggregate.cir_ci95 is not None for aggregate in run.aggregates)
    contextos = next(aggregate for aggregate in run.aggregates if aggregate.strategy == "contextos")
    assert contextos.successful_case_count == 50
    assert contextos.mean_task_specific_score >= 0.0
    assert contextos.mean_critical_information_recall >= 0.0
