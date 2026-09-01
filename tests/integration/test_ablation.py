"""End-to-end Phase 4E ablation-study acceptance tests."""

from pathlib import Path

import pytest

from contextos.benchmarks.artifacts import load_dataset
from contextos.benchmarks.runner import (
    ablation_effects,
    default_ablation_strategies,
    run_contextos_bench,
)
from contextos.tokenization import TiktokenTokenizer

EXPECTED_ABLATIONS = [
    "contextos_full",
    "contextos_without_semantic_deduplication",
    "contextos_without_recency",
    "contextos_without_dependency_score",
    "contextos_without_compression",
    "contextos_without_position_aware_layout",
]


def test_ablation_study_runs_every_variant_on_every_case() -> None:
    dataset = load_dataset(Path("benchmarks/datasets/contextos_bench.json"))
    run = run_contextos_bench(
        dataset,
        tokenizer=TiktokenTokenizer(),
        strategies=default_ablation_strategies(),
    )

    assert run.strategies == EXPECTED_ABLATIONS
    assert len(run.measurements) == 300
    assert {aggregate.strategy for aggregate in run.aggregates} == set(EXPECTED_ABLATIONS)
    assert all(aggregate.case_count == 50 for aggregate in run.aggregates)
    assert all(aggregate.successful_case_count == 50 for aggregate in run.aggregates)
    assert all(aggregate.task_score_ci95 is not None for aggregate in run.aggregates)
    assert all(aggregate.cir_ci95 is not None for aggregate in run.aggregates)
    assert all(aggregate.mean_input_tokens >= 0 for aggregate in run.aggregates)
    assert all(aggregate.p95_optimizer_latency_ms >= 0 for aggregate in run.aggregates)
    assert run.metadata["strategy_configurations"] == {
        strategy.name: strategy.policy_overrides for strategy in default_ablation_strategies()
    }
    effects = ablation_effects(run)
    assert effects["contextos_without_dependency_score"]["task_score_delta"] == pytest.approx(-0.08)
    assert effects["contextos_without_dependency_score"]["cir_delta"] == 0.0


def test_each_ablation_changes_exactly_one_policy_component() -> None:
    strategies = default_ablation_strategies()
    overrides = [strategy.policy_overrides for strategy in strategies]

    assert overrides == [
        {},
        {"semantic_dedup_enabled": False},
        {"weight_recency": 0.0},
        {"weight_dependency": 0.0},
        {"compression_enabled": False},
        {"position_aware_layout": False},
    ]


def test_ablation_effects_require_full_contextos_reference() -> None:
    dataset = load_dataset(Path("benchmarks/datasets/contextos_bench.json"))
    run = run_contextos_bench(
        dataset,
        tokenizer=TiktokenTokenizer(),
        strategies=default_ablation_strategies()[1:],
        case_limit=1,
    )

    with pytest.raises(ValueError, match="contextos_full"):
        ablation_effects(run)
