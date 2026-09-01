"""End-to-end offline positional experiment acceptance test."""

from contextos.benchmarks.positional import (
    build_positional_dataset,
    run_positional_benchmark,
)
from contextos.providers import DeterministicRetrievalProvider
from contextos.tokenization import TiktokenTokenizer


def test_positional_experiment_runs_every_layout_and_aggregation_cell() -> None:
    dataset = build_positional_dataset(context_lengths=[256, 512])
    run = run_positional_benchmark(
        dataset,
        provider=DeterministicRetrievalProvider(),
        provider_name="deterministic",
        provider_model="deterministic-retrieval-v1",
        tokenizer=TiktokenTokenizer(),
        profile="integration",
        max_context_tokens=544,
    )

    assert len(run.predictions) == 30
    assert len(run.accuracy_cells) == 30
    assert len(run.robustness) == 6
    assert run.executed_context_lengths == [256, 512]
    assert run.skipped_context_lengths == []
    assert all(prediction.exact_match for prediction in run.predictions)
