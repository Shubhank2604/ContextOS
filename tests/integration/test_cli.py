"""CLI integration tests."""

import json
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from contextos.benchmarks.models import BenchmarkRun
from contextos.benchmarks.positional_models import PositionalRun
from contextos.cli import app
from contextos.models import ContextItem, ContextType

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Construct and inspect LLM context" in result.stdout


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.3.0"


def write_input(path: Path) -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    item = ContextItem(
        id="cli-item",
        content="A compact CLI fixture.",
        type=ContextType.USER_MESSAGE,
        created_at=timestamp,
        updated_at=timestamp,
        importance=1.0,
    )
    path.write_text(json.dumps([item.model_dump(mode="json")]), encoding="utf-8")


def test_cli_optimize_writes_json_trace(tmp_path: Path) -> None:
    input_path = tmp_path / "items.json"
    trace_path = tmp_path / "trace.json"
    write_input(input_path)

    result = runner.invoke(
        app,
        [
            "optimize",
            "--input",
            str(input_path),
            "--budget",
            "20",
            "--strategy",
            "full",
            "--trace-json",
            str(trace_path),
        ],
    )
    assert result.exit_code == 0
    assert "Selected items: cli-item" in result.stdout
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["strategy"] == "full_context"
    assert trace["items"][0]["decision"] == "retained"


def test_cli_returns_nonzero_for_full_context_overflow(tmp_path: Path) -> None:
    input_path = tmp_path / "items.json"
    write_input(input_path)
    result = runner.invoke(
        app,
        ["optimize", "--input", str(input_path), "--budget", "1", "--strategy", "full"],
    )
    assert result.exit_code == 2
    assert "Optimization failed" in result.stderr
    assert "effective budget" in result.stderr


def test_cli_quick_benchmark_runs_end_to_end() -> None:
    result = runner.invoke(app, ["benchmark", "--profile", "quick"])
    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["profile"] == "quick"
    assert len(report["results"]) == 4
    assert report["results"][-1]["strategy"] == "contextos"


def test_cli_contextos_optimize_is_the_default(tmp_path: Path) -> None:
    input_path = tmp_path / "items.json"
    write_input(input_path)
    result = runner.invoke(
        app,
        ["optimize", "--input", str(input_path), "--budget", "20", "--task", "fixture"],
    )
    assert result.exit_code == 0
    assert "Strategy: contextos" in result.stdout


def test_cli_inspect_runs_without_optimization(tmp_path: Path) -> None:
    input_path = tmp_path / "items.json"
    write_input(input_path)
    result = runner.invoke(app, ["inspect", "--input", str(input_path)])
    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["item_count"] == 1
    assert report["items_by_type"] == {"user_message": 1}


def test_cli_deduplication_benchmark_runs_end_to_end() -> None:
    result = runner.invoke(
        app,
        [
            "benchmark",
            "dedup",
            "--input",
            "benchmarks/datasets/deduplication_cases.json",
        ],
    )
    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["case_count"] == 10
    assert report["false_positive"] == 0
    assert report["f1"] == 1.0


def test_cli_rejects_unknown_benchmark_profile() -> None:
    result = runner.invoke(app, ["benchmark", "--profile", "standard"])
    assert result.exit_code == 2
    assert "supports only 'quick'" in result.stderr


def test_cli_contextos_bench_writes_immutable_artifact(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "benchmark",
            "run",
            "--input",
            "benchmarks/datasets/contextos_bench.json",
            "--output-directory",
            str(tmp_path),
            "--case-limit",
            "1",
        ],
    )

    assert result.exit_code == 0
    report = json.loads(result.stdout)
    artifact = Path(report["artifact"])
    assert artifact.exists()
    run = BenchmarkRun.model_validate_json(artifact.read_text(encoding="utf-8"))
    assert run.metadata["case_count"] == 1
    assert len(run.measurements) == 4


def test_cli_benchmark_compare_reports_zero_delta_for_same_run(tmp_path: Path) -> None:
    run_result = runner.invoke(
        app,
        [
            "benchmark",
            "run",
            "--output-directory",
            str(tmp_path),
            "--case-limit",
            "1",
        ],
    )
    artifact = json.loads(run_result.stdout)["artifact"]
    comparison = runner.invoke(
        app,
        ["benchmark", "compare", "--left", artifact, "--right", artifact],
    )

    assert comparison.exit_code == 0
    report = json.loads(comparison.stdout)
    assert report["contextos"]["task_score_delta"] == 0.0
    assert report["contextos"]["cir_delta"] == 0.0


def test_cli_positional_quick_run_writes_raw_artifact(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "benchmark",
            "positional",
            "--input",
            "benchmarks/datasets/positional_retrieval.json",
            "--output-directory",
            str(tmp_path),
            "--profile",
            "quick",
            "--provider",
            "deterministic",
        ],
    )

    assert result.exit_code == 0
    report = json.loads(result.stdout)
    artifact = Path(report["artifact"])
    run = PositionalRun.model_validate_json(artifact.read_text(encoding="utf-8"))
    assert report["prediction_count"] == 15
    assert run.executed_context_lengths == [4_096]
    assert len(run.robustness) == 3


def test_cli_positional_openai_requires_explicit_model(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "benchmark",
            "positional",
            "--output-directory",
            str(tmp_path),
            "--provider",
            "openai",
        ],
    )

    assert result.exit_code == 2
    assert "--model is required" in result.stderr
