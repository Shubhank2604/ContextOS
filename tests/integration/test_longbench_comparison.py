"""Offline six-strategy LongBench comparison integration tests."""

from contextos.benchmarks.longbench import score_longbench_predictions
from contextos.benchmarks.longbench_models import (
    LongBenchCase,
    LongBenchMetric,
    LongBenchProfile,
    PreparedLongBenchSubset,
)
from contextos.benchmarks.longbench_runner import (
    chunk_longbench_context,
    run_longbench_comparison,
)
from contextos.providers.base import ProviderResponse


class WordTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text.split())


class FixedAnswerProvider:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    def complete(self, prompt: str, *, max_output_tokens: int) -> ProviderResponse:
        self.calls += 1
        return ProviderResponse(
            text=self.answer,
            input_tokens=len(prompt.split()),
            output_tokens=min(len(self.answer.split()), max_output_tokens),
            model="fixture-v1",
        )


def _subset() -> PreparedLongBenchSubset:
    case = LongBenchCase(
        dataset="hotpotqa",
        source_id="external-001",
        input="Which timeout is approved?",
        context=(
            "Formatting completed successfully. "
            "The approved authentication timeout is thirty seconds. "
            "A stale note proposed sixty seconds."
        ),
        answers=["thirty seconds"],
        source_length=15,
        language="en",
        metric=LongBenchMetric.QA_F1,
        prompt_template="Context: {context}\nQuestion: {input}\nAnswer:",
        max_output_tokens=4,
    )
    return PreparedLongBenchSubset(
        profile=LongBenchProfile.QUICK,
        source_repository="fixture",
        source_revision="fixture-revision",
        source_split="test",
        sampling_seed=1,
        cases=[case],
    )


def test_chunking_preserves_external_context_exactly() -> None:
    case = _subset().cases[0]
    chunks = chunk_longbench_context(case, tokenizer=WordTokenizer(), max_chunk_tokens=4)

    assert len(chunks) > 1
    assert "".join(chunk.content for chunk in chunks) == case.context
    assert all(chunk.token_count is None for chunk in chunks)
    assert [chunk.metadata["character_start"] for chunk in chunks] == sorted(
        chunk.metadata["character_start"] for chunk in chunks
    )


def test_six_strategies_share_provider_model_cases_and_evaluator() -> None:
    subset = _subset()
    provider = FixedAnswerProvider("thirty seconds")
    predictions = run_longbench_comparison(
        subset,
        provider=provider,
        provider_name="fixture",
        provider_model="fixture-v1",
        tokenizer=WordTokenizer(),
        context_budget_tokens=8,
        max_context_tokens=100,
        max_chunk_tokens=4,
    )
    report = score_longbench_predictions(subset, predictions)

    assert provider.calls == 6
    assert {prediction.strategy for prediction in predictions} == {
        "contextos",
        "full_context",
        "last_n",
        "naive_extractive",
        "relevance_only",
        "sliding_window",
    }
    assert all(prediction.provider == "fixture" for prediction in predictions)
    assert all(prediction.model == "fixture-v1" for prediction in predictions)
    assert all(prediction.status == "ok" for prediction in predictions)
    assert all(prediction.prompt_input_tokens is not None for prediction in predictions)
    assert len(report.case_scores) == 6
    assert len(report.dataset_aggregates) == 6
    assert all(score.score == 1.0 for score in report.case_scores)
    assert all(score.quality_retention == 1.0 for score in report.case_scores)


def test_full_context_infeasibility_is_raw_and_not_scored() -> None:
    subset = _subset()
    provider = FixedAnswerProvider("thirty seconds")
    predictions = run_longbench_comparison(
        subset,
        provider=provider,
        provider_name="fixture",
        provider_model="fixture-v1",
        tokenizer=WordTokenizer(),
        context_budget_tokens=3,
        max_context_tokens=16,
        max_chunk_tokens=3,
    )
    report = score_longbench_predictions(subset, predictions)
    full_prediction = next(
        prediction for prediction in predictions if prediction.strategy == "full_context"
    )
    full_score = next(score for score in report.case_scores if score.strategy == "full_context")

    assert full_prediction.status == "context_overflow"
    assert full_score.score is None
    assert full_score.quality_retention is None
