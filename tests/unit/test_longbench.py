"""Offline LongBench adapter, sampling, metric, and artifact tests."""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest

from contextos.benchmarks.longbench import (
    code_similarity_score,
    load_longbench_config,
    prepare_longbench_subset,
    qa_f1_score,
    render_longbench_prompt,
    retrieval_score,
    score_longbench_predictions,
    write_prepared_subset,
    write_score_report,
)
from contextos.benchmarks.longbench_models import (
    LongBenchPrediction,
    LongBenchProfile,
    LongBenchSubsetConfig,
)


class FakeLongBenchSource:
    """Return deterministic official-shaped rows without network access."""

    def __init__(self, *, reverse: bool = False, rows_per_task: int = 30) -> None:
        self.reverse = reverse
        self.rows_per_task = rows_per_task
        self.calls: list[tuple[str, str, str, str]] = []

    def load_task(
        self,
        repository: str,
        dataset: str,
        *,
        split: str,
        revision: str,
    ) -> Iterable[Mapping[str, Any]]:
        self.calls.append((repository, dataset, split, revision))
        rows = [
            {
                "_id": f"{dataset}-{index:03d}",
                "input": f"Question {index}?",
                "context": f"Context for {dataset} example {index}.",
                "answers": (
                    [f"Paragraph {index}"]
                    if dataset == "passage_retrieval_en"
                    else [f"answer {index}"]
                ),
                "length": 10 + index,
                "language": "en",
                "all_classes": None,
            }
            for index in range(self.rows_per_task)
        ]
        return reversed(rows) if self.reverse else rows


def _config() -> LongBenchSubsetConfig:
    return load_longbench_config(Path("benchmarks/config/longbench_subset.json"))


def test_config_pins_representative_tasks_and_standard_scale() -> None:
    config = _config()

    assert config.source_repository == "zai-org/LongBench"
    assert config.source_revision != "main"
    assert {task.dataset for task in config.tasks} == {
        "hotpotqa",
        "2wikimqa",
        "passage_retrieval_en",
        "repobench-p",
    }
    assert sum(task.standard_samples for task in config.tasks) == 100
    assert all("{context}" in task.prompt_template for task in config.tasks)
    assert all("{input}" in task.prompt_template for task in config.tasks)


def test_profiles_preserve_ids_and_sample_independently_of_source_order() -> None:
    config = _config()
    forward = prepare_longbench_subset(
        config,
        profile=LongBenchProfile.QUICK,
        source=FakeLongBenchSource(),
    )
    reverse = prepare_longbench_subset(
        config,
        profile=LongBenchProfile.QUICK,
        source=FakeLongBenchSource(reverse=True),
    )
    standard = prepare_longbench_subset(
        config,
        profile=LongBenchProfile.STANDARD,
        source=FakeLongBenchSource(),
    )
    full = prepare_longbench_subset(
        config,
        profile=LongBenchProfile.FULL,
        source=FakeLongBenchSource(),
    )

    assert forward == reverse
    assert len(forward.cases) == 8
    assert len(standard.cases) == 100
    assert len(full.cases) == 120
    assert all(case.source_id.startswith(case.dataset) for case in forward.cases)
    assert "fixture" not in render_longbench_prompt(forward.cases[0])
    assert "optimized context" in render_longbench_prompt(
        forward.cases[0], context="optimized context"
    )


def test_dataset_appropriate_metrics_match_longbench_behavior() -> None:
    assert qa_f1_score("The Eiffel, Tower", "eiffel tower") == 1.0
    assert qa_f1_score("alpha beta", "alpha gamma") == pytest.approx(0.5)
    assert retrieval_score("Paragraph 4 or Paragraph 7", "Paragraph 7") == 0.5
    assert retrieval_score("unknown", "Paragraph 7") == 0.0
    assert code_similarity_score("# explanation\nreturn value", "return value") == 1.0
    with pytest.raises(ValueError, match="lacks a paragraph ID"):
        retrieval_score("Paragraph 2", "invalid")


def test_scoring_joins_preserved_ids_and_keeps_metrics_separate() -> None:
    subset = prepare_longbench_subset(
        _config(),
        profile=LongBenchProfile.QUICK,
        source=FakeLongBenchSource(),
    )
    predictions = [
        LongBenchPrediction(
            dataset=case.dataset,
            source_id=case.source_id,
            prediction=case.answers[0],
            provider="fixture",
            model="fixture-v1",
        )
        for case in subset.cases
    ]
    report = score_longbench_predictions(subset, predictions)

    assert report.prediction_count == 8
    assert len(report.case_scores) == 8
    assert len(report.dataset_aggregates) == 4
    assert all(aggregate.mean_score == 1.0 for aggregate in report.dataset_aggregates)
    assert {aggregate.metric for aggregate in report.dataset_aggregates} == {
        case.metric for case in subset.cases
    }

    with pytest.raises(ValueError, match="identity mismatch"):
        score_longbench_predictions(subset, predictions[:-1])
    mixed = [*predictions[:-1], predictions[-1].model_copy(update={"model": "other"})]
    with pytest.raises(ValueError, match="one provider and model"):
        score_longbench_predictions(subset, mixed)


def test_prepared_and_score_outputs_refuse_conflicting_overwrites(tmp_path: Path) -> None:
    subset = prepare_longbench_subset(
        _config(),
        profile=LongBenchProfile.QUICK,
        source=FakeLongBenchSource(),
    )
    prepared_path = tmp_path / "prepared.json"
    assert write_prepared_subset(subset, prepared_path) == prepared_path
    assert write_prepared_subset(subset, prepared_path) == prepared_path
    prepared_path.write_text(json.dumps({"different": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="different content"):
        write_prepared_subset(subset, prepared_path)

    predictions = [
        LongBenchPrediction(
            dataset=case.dataset,
            source_id=case.source_id,
            prediction=case.answers[0],
            provider="fixture",
            model="fixture-v1",
        )
        for case in subset.cases
    ]
    report = score_longbench_predictions(subset, predictions)
    score_path = tmp_path / "scores.json"
    assert write_score_report(report, score_path) == score_path
    assert write_score_report(report, score_path) == score_path
