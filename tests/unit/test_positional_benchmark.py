"""Controlled positional-retrieval generation, evaluation, and artifact tests."""

from pathlib import Path

import pytest

from contextos.benchmarks.positional import (
    REQUIRED_CONTEXT_LENGTHS,
    aggregate_positional_predictions,
    build_positional_dataset,
    construct_positional_prompt,
    load_positional_dataset,
    run_positional_benchmark,
    write_positional_run_artifact,
)
from contextos.benchmarks.positional_models import (
    EvidencePosition,
    PositionalPrediction,
    PositionalStrategy,
)
from contextos.providers import DeterministicRetrievalProvider
from contextos.tokenization import TiktokenTokenizer


def test_canonical_positional_grid_is_reproducible_and_complete() -> None:
    canonical = load_positional_dataset(Path("benchmarks/datasets/positional_retrieval.json"))

    assert canonical == build_positional_dataset()
    assert len(canonical.cases) == 20
    assert {case.target_context_tokens for case in canonical.cases} == set(REQUIRED_CONTEXT_LENGTHS)
    assert {case.evidence_position for case in canonical.cases} == set(EvidencePosition)


@pytest.mark.parametrize("position", list(EvidencePosition))
def test_original_prompt_hits_requested_token_position(position: EvidencePosition) -> None:
    tokenizer = TiktokenTokenizer()
    case = next(
        case
        for case in build_positional_dataset(context_lengths=[4_096]).cases
        if case.evidence_position is position
    )
    prompt, distractor_count, actual_fraction = construct_positional_prompt(
        case,
        strategy=PositionalStrategy.ORIGINAL_FULL,
        tokenizer=tokenizer,
    )

    assert distractor_count > 100
    assert 4_040 <= tokenizer.count_tokens(prompt) <= 4_096
    assert actual_fraction == pytest.approx(position.fraction, abs=0.02)


def test_optimized_layouts_move_middle_evidence_to_the_front() -> None:
    tokenizer = TiktokenTokenizer()
    case = next(
        case
        for case in build_positional_dataset(context_lengths=[4_096]).cases
        if case.evidence_position is EvidencePosition.MIDDLE
    )
    _, distractor_count, original_fraction = construct_positional_prompt(
        case,
        strategy=PositionalStrategy.ORIGINAL_FULL,
        tokenizer=tokenizer,
    )
    relevance_prompt, _, relevance_fraction = construct_positional_prompt(
        case,
        strategy=PositionalStrategy.RELEVANCE_DESCENDING,
        tokenizer=tokenizer,
        distractor_count=distractor_count,
    )
    contextos_prompt, _, contextos_fraction = construct_positional_prompt(
        case,
        strategy=PositionalStrategy.CONTEXTOS_POSITION_AWARE,
        tokenizer=tokenizer,
        distractor_count=distractor_count,
    )

    assert original_fraction == pytest.approx(0.5, abs=0.02)
    assert relevance_fraction < 0.02
    assert contextos_fraction < 0.02
    assert relevance_prompt == contextos_prompt


def test_offline_runner_exact_matches_and_enforces_provider_limit() -> None:
    dataset = build_positional_dataset(context_lengths=[256, 512])
    run = run_positional_benchmark(
        dataset,
        provider=DeterministicRetrievalProvider(),
        provider_name="deterministic",
        provider_model="deterministic-retrieval-v1",
        tokenizer=TiktokenTokenizer(),
        profile="test",
        max_context_tokens=288,
    )

    assert run.executed_context_lengths == [256]
    assert run.skipped_context_lengths == [512]
    assert len(run.predictions) == 15
    assert all(prediction.exact_match for prediction in run.predictions)
    assert all(metric.max_min_positional_gap == 0.0 for metric in run.robustness)


def _raw_prediction(position: EvidencePosition, *, exact_match: bool) -> PositionalPrediction:
    return PositionalPrediction(
        case_id=f"case-{position.value}",
        strategy=PositionalStrategy.ORIGINAL_FULL,
        target_context_tokens=4_096,
        evidence_position=position,
        actual_evidence_fraction=position.fraction,
        distractor_count=100,
        expected_value="EXPECTED",
        raw_prediction="EXPECTED" if exact_match else "WRONG",
        normalized_prediction="expected" if exact_match else "wrong",
        exact_match=exact_match,
        estimated_input_tokens=4_090,
        model_total_latency_ms=10.0,
        prompt_sha256="a" * 64,
    )


def test_aggregation_reports_position_gap_variance_and_tokens() -> None:
    predictions = [
        _raw_prediction(position, exact_match=position is not EvidencePosition.MIDDLE)
        for position in EvidencePosition
    ]
    cells, robustness = aggregate_positional_predictions(predictions)

    assert len(cells) == 5
    assert robustness[0].accuracy_by_position[EvidencePosition.MIDDLE] == 0.0
    assert robustness[0].max_min_positional_gap == 1.0
    assert robustness[0].positional_variance == pytest.approx(0.16)
    assert robustness[0].positional_std_dev == pytest.approx(0.4)


def test_positional_artifacts_are_immutable(tmp_path: Path) -> None:
    dataset = build_positional_dataset(context_lengths=[256])
    run = run_positional_benchmark(
        dataset,
        provider=DeterministicRetrievalProvider(),
        provider_name="deterministic",
        provider_model="deterministic-retrieval-v1",
        tokenizer=TiktokenTokenizer(),
        profile="test",
        max_context_tokens=288,
        strategies=[PositionalStrategy.ORIGINAL_FULL],
    )
    path = write_positional_run_artifact(run, tmp_path)

    assert path == write_positional_run_artifact(run, tmp_path)
    conflicting = run.model_copy(update={"provider": "different"})
    with pytest.raises(ValueError, match="collision"):
        write_positional_run_artifact(conflicting, tmp_path)
