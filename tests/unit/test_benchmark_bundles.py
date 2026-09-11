"""Tests for the shared immutable Phase 4F bundle writer."""

import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from contextos.benchmarks.bundles import (
    REQUIRED_BUNDLE_FILES,
    BenchmarkEnvironment,
    capture_environment,
    load_benchmark_bundle,
    write_benchmark_bundle,
)

RECORDED_AT = datetime(2026, 9, 11, 12, 30, tzinfo=UTC)


def environment(recorded_at: datetime = RECORDED_AT) -> BenchmarkEnvironment:
    return BenchmarkEnvironment(
        recorded_at_utc=recorded_at,
        python_version="3.13.0",
        contextos_version="0.3.0",
        git_sha="abc123",
        operating_system="fixture-os",
        dependency_versions={"pydantic": "2.0"},
        embedding_provider="fixture",
        embedding_model="fixture-embedding",
    )


def write_fixture_bundle(
    output_directory: Path,
    *,
    report: str = "# Fixture report",
    cases: list[dict[str, str]] | None = None,
    predictions: list[dict[str, str | float]] | None = None,
    manifest: BenchmarkEnvironment | None = None,
) -> Path:
    return write_benchmark_bundle(
        output_directory=output_directory,
        recorded_at_utc=RECORDED_AT,
        strategy="ContextOS comparison",
        profile="quick",
        config={"benchmark": "fixture"},
        environment=manifest or environment(),
        cases=[{"id": "case-1"}] if cases is None else cases,
        predictions=([{"case_id": "case-1", "score": 1.0}] if predictions is None else predictions),
        metrics={"mean_score": 1.0},
        metric_rows=[{"strategy": "contextos", "nested": {"low": 0.9}}],
        report=report,
    )


def test_bundle_contains_exact_required_files_and_is_idempotent(tmp_path: Path) -> None:
    path = write_fixture_bundle(tmp_path)

    assert path == write_fixture_bundle(tmp_path)
    assert path.name == "20260911T123000.000000Z-contextos-comparison-quick"
    assert {entry.name for entry in path.iterdir()} == REQUIRED_BUNDLE_FILES
    loaded = load_benchmark_bundle(path)
    assert loaded.cases[0]["id"] == "case-1"
    assert loaded.environment.git_sha == "abc123"
    assert loaded.metrics == {"mean_score": 1.0}
    with (path / "metrics.csv").open(encoding="utf-8", newline="") as source:
        assert next(csv.DictReader(source))["nested"] == '{"low":0.9}'


def test_bundle_refuses_content_collision_and_modified_file_set(tmp_path: Path) -> None:
    path = write_fixture_bundle(tmp_path)
    with pytest.raises(ValueError, match="collision"):
        write_fixture_bundle(tmp_path, report="# Different")

    (path / "unexpected.txt").write_text("mutation", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete or modified"):
        write_fixture_bundle(tmp_path)


def test_bundle_validates_required_records_and_matching_timestamp(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one case"):
        write_fixture_bundle(tmp_path, cases=[])
    with pytest.raises(ValueError, match="at least one prediction"):
        write_fixture_bundle(tmp_path, predictions=[])
    with pytest.raises(ValueError, match="timestamp"):
        write_fixture_bundle(
            tmp_path,
            manifest=environment(RECORDED_AT + timedelta(seconds=1)),
        )


def test_bundle_rejects_empty_metrics_and_report(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="metric row"):
        write_benchmark_bundle(
            output_directory=tmp_path,
            recorded_at_utc=RECORDED_AT,
            strategy="fixture",
            profile="quick",
            config={"benchmark": "fixture"},
            environment=environment(),
            cases=[{"id": "case-1"}],
            predictions=[{"case_id": "case-1"}],
            metrics={},
            metric_rows=[],
            report="# Fixture",
        )
    with pytest.raises(ValueError, match="report"):
        write_benchmark_bundle(
            output_directory=tmp_path,
            recorded_at_utc=RECORDED_AT,
            strategy="fixture",
            profile="quick",
            config={"benchmark": "fixture"},
            environment=environment(),
            cases=[{"id": "case-1"}],
            predictions=[{"case_id": "case-1"}],
            metrics={},
            metric_rows=[{"score": 1.0}],
            report=" ",
        )


def test_environment_capture_records_required_provenance() -> None:
    result = capture_environment(
        recorded_at_utc=RECORDED_AT,
        embedding_provider="contextos",
        embedding_model="fixture",
        llm_provider="fixture-provider",
        llm_model="fixture-model",
        dependency_names=("definitely-not-installed-contextos-fixture",),
    )

    assert result.recorded_at_utc == RECORDED_AT
    assert result.python_version
    assert result.contextos_version
    assert result.git_sha
    assert result.operating_system
    assert result.dependency_versions["definitely-not-installed-contextos-fixture"] == (
        "not-installed"
    )
