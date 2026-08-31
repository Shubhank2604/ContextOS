"""Tests for labeled deduplication quality metrics."""

import json
from pathlib import Path

from pydantic import TypeAdapter

from contextos.dedup.metrics import DeduplicationCase, evaluate_deduplication_cases
from contextos.embeddings import DeterministicEmbeddingProvider


def test_labeled_deduplication_fixture_reports_raw_and_derived_metrics() -> None:
    fixture_path = Path("benchmarks/datasets/deduplication_cases.json")
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    cases = TypeAdapter(list[DeduplicationCase]).validate_python(payload)
    metrics = evaluate_deduplication_cases(
        cases,
        provider=DeterministicEmbeddingProvider(),
    )
    assert metrics.case_count == len(cases) == 10
    assert metrics.false_positive == 0
    assert metrics.true_positive >= 2
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
