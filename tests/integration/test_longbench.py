"""End-to-end offline LongBench subset integration test."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from contextos.benchmarks.longbench import (
    load_longbench_config,
    prepare_longbench_subset,
    score_longbench_predictions,
)
from contextos.benchmarks.longbench_models import (
    LongBenchPrediction,
    LongBenchProfile,
)


class IntegrationSource:
    def load_task(
        self,
        repository: str,
        dataset: str,
        *,
        split: str,
        revision: str,
    ) -> Iterable[Mapping[str, Any]]:
        del repository, split, revision
        return [
            {
                "_id": f"{dataset}-{index:03d}",
                "input": "Return the fixture answer.",
                "context": "A synthetic integration fixture, not LongBench data.",
                "answers": [
                    f"Paragraph {index}" if dataset == "passage_retrieval_en" else "fixture answer"
                ],
                "length": 8,
                "language": "en",
                "all_classes": None,
            }
            for index in range(25)
        ]


def test_standard_profile_materializes_and_scores_100_external_shaped_cases() -> None:
    config = load_longbench_config(Path("benchmarks/config/longbench_subset.json"))
    subset = prepare_longbench_subset(
        config,
        profile=LongBenchProfile.STANDARD,
        source=IntegrationSource(),
    )
    predictions = [
        LongBenchPrediction(
            dataset=case.dataset,
            source_id=case.source_id,
            prediction=case.answers[0],
            provider="integration",
            model="integration-v1",
        )
        for case in subset.cases
    ]
    report = score_longbench_predictions(subset, predictions)

    assert len(subset.cases) == 100
    assert report.prediction_count == 100
    assert len(report.dataset_aggregates) == 4
    assert all(aggregate.case_count == 25 for aggregate in report.dataset_aggregates)
    assert all(aggregate.mean_score == 1.0 for aggregate in report.dataset_aggregates)
