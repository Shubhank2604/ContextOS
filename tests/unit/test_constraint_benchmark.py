"""Phase 5B constraint-sensitive benchmark tests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from contextos.benchmarks.bundles import REQUIRED_BUNDLE_FILES
from contextos.benchmarks.constraint_benchmark import (
    run_constraint_benchmark,
    write_constraint_artifact,
)
from contextos.benchmarks.constraint_dataset import (
    PHASE5_BASELINE_SHA,
    generate_constraint_dataset,
)
from contextos.benchmarks.constraint_models import (
    ConstraintBenchmarkCase,
    ConstraintCategory,
)
from contextos.benchmarks.runner import ContextOSBenchmarkStrategy
from contextos.tokenization import TiktokenTokenizer


def test_constraint_dataset_has_ten_cases_per_required_category() -> None:
    dataset = generate_constraint_dataset()

    assert dataset.name == "ContextOS-Bench Constraints"
    assert len(dataset.cases) == 90
    assert Counter(case.constraints.category for case in dataset.cases) == {
        category: 10 for category in ConstraintCategory
    }
    assert all(case.constraints.critical_item_ids for case in dataset.cases)
    assert all(case.answer_key.required_facts for case in dataset.cases)


def test_constraint_annotations_reject_unknown_item_references() -> None:
    case = generate_constraint_dataset().cases[0]
    payload = case.model_dump()
    payload["constraints"]["critical_item_ids"] = ["missing-item"]

    with pytest.raises(ValidationError, match="unknown items"):
        ConstraintBenchmarkCase.model_validate(payload)


def test_constraint_runner_reports_every_metric_family() -> None:
    dataset = generate_constraint_dataset()
    representative = dataset.model_copy(
        update={"cases": [dataset.cases[index] for index in range(0, 90, 10)]}
    )

    run, measurements, aggregates = run_constraint_benchmark(
        representative,
        tokenizer=TiktokenTokenizer(),
        strategies=[ContextOSBenchmarkStrategy()],
    )

    assert len(run.measurements) == 9
    assert len(measurements) == 9
    assert {measurement.category for measurement in measurements} == set(ConstraintCategory)
    assert len(aggregates) == 1
    aggregate = aggregates[0]
    assert aggregate.case_count == 9
    assert aggregate.dependency_closure_rate is not None
    assert aggregate.state_consistency_rate is not None
    assert aggregate.contradiction_leakage_rate is not None
    assert aggregate.exact_value_preservation_rate is not None
    assert aggregate.identifier_preservation_rate is not None
    assert aggregate.citation_preservation_rate is not None


def test_constraint_artifact_records_frozen_baseline(tmp_path: Path) -> None:
    dataset = generate_constraint_dataset()
    limited = dataset.model_copy(update={"cases": dataset.cases[:1]})
    run, measurements, aggregates = run_constraint_benchmark(
        limited,
        tokenizer=TiktokenTokenizer(),
        strategies=[ContextOSBenchmarkStrategy()],
    )

    artifact = write_constraint_artifact(
        tmp_path,
        dataset=limited,
        run=run,
        measurements=measurements,
        aggregates=aggregates,
        profile="test",
    )

    assert {entry.name for entry in artifact.iterdir()} == REQUIRED_BUNDLE_FILES
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((artifact / "metrics.json").read_text(encoding="utf-8"))
    assert config["baseline_v040_sha"] == PHASE5_BASELINE_SHA
    assert metrics["baseline_v040_sha"] == PHASE5_BASELINE_SHA
    assert config["current_research_sha"]
