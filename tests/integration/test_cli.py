"""CLI integration tests."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from contextos.benchmarks.longbench_models import (
    LongBenchCase,
    LongBenchMetric,
    LongBenchPrediction,
    LongBenchProfile,
    PreparedLongBenchSubset,
)
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
    assert len(report["results"]) == 6
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


@pytest.mark.parametrize("strategy", ["relevance-only", "naive-extractive"])
def test_cli_exposes_complete_phase4d_baselines(tmp_path: Path, strategy: str) -> None:
    input_path = tmp_path / "items.json"
    write_input(input_path)
    result = runner.invoke(
        app,
        [
            "optimize",
            "--input",
            str(input_path),
            "--budget",
            "20",
            "--task",
            "compact fixture",
            "--strategy",
            strategy,
        ],
    )

    assert result.exit_code == 0
    assert "Selected items: cli-item" in result.stdout


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
    assert len(run.measurements) == 6


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


def test_cli_longbench_scores_complete_id_keyed_predictions(tmp_path: Path) -> None:
    case = LongBenchCase(
        dataset="hotpotqa",
        source_id="source-1",
        input="What is the answer?",
        context="The answer is ContextOS.",
        answers=["ContextOS"],
        source_length=4,
        language="en",
        metric=LongBenchMetric.QA_F1,
        prompt_template="Context: {context}\nQuestion: {input}\nAnswer:",
        max_output_tokens=32,
    )
    subset = PreparedLongBenchSubset(
        profile=LongBenchProfile.QUICK,
        source_repository="fixture",
        source_revision="fixture-revision",
        source_split="test",
        sampling_seed=1,
        cases=[case],
    )
    prepared_path = tmp_path / "prepared.json"
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "scores.json"
    prepared_path.write_text(subset.model_dump_json(indent=2), encoding="utf-8")
    prediction = LongBenchPrediction(
        dataset=case.dataset,
        source_id=case.source_id,
        prediction="ContextOS",
        provider="fixture",
        model="fixture-v1",
    )
    predictions_path.write_text(prediction.model_dump_json() + "\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "benchmark",
            "longbench",
            "score",
            "--prepared",
            str(prepared_path),
            "--predictions",
            str(predictions_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["prediction_count"] == 1
    assert report["dataset_aggregates"][0]["mean_score"] == 1.0


def test_cli_longbench_prepare_uses_explicit_external_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load_task(
        self: object,
        repository: str,
        dataset: str,
        *,
        split: str,
        revision: str,
    ) -> list[dict[str, object]]:
        del self, repository, split, revision
        return [
            {
                "_id": f"{dataset}-{index}",
                "input": "fixture question",
                "context": "fixture context",
                "answers": [
                    f"Paragraph {index}" if dataset == "passage_retrieval_en" else "fixture"
                ],
                "length": 2,
                "language": "en",
                "all_classes": None,
            }
            for index in range(2)
        ]

    monkeypatch.setattr(
        "contextos.cli.HuggingFaceLongBenchSource.load_task",
        fake_load_task,
    )
    output_path = tmp_path / "prepared.json"
    result = runner.invoke(
        app,
        [
            "benchmark",
            "longbench",
            "prepare",
            "--profile",
            "quick",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["case_count"] == 8
    assert report["cases_by_dataset"] == {
        "2wikimqa": 2,
        "hotpotqa": 2,
        "passage_retrieval_en": 2,
        "repobench-p": 2,
    }


def test_cli_longbench_run_requires_working_explicit_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    case = LongBenchCase(
        dataset="hotpotqa",
        source_id="source-1",
        input="What is the answer?",
        context="The answer is ContextOS.",
        answers=["ContextOS"],
        source_length=4,
        language="en",
        metric=LongBenchMetric.QA_F1,
        prompt_template="Context: {context}\nQuestion: {input}\nAnswer:",
        max_output_tokens=4,
    )
    subset = PreparedLongBenchSubset(
        profile=LongBenchProfile.QUICK,
        source_repository="fixture",
        source_revision="fixture-revision",
        source_split="test",
        sampling_seed=1,
        cases=[case],
    )
    prepared_path = tmp_path / "prepared.json"
    output_path = tmp_path / "predictions.jsonl"
    prepared_path.write_text(subset.model_dump_json(indent=2), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "benchmark",
            "longbench",
            "run",
            "--prepared",
            str(prepared_path),
            "--output",
            str(output_path),
            "--model",
            "configured-model",
            "--context-budget-tokens",
            "16",
            "--max-context-tokens",
            "128",
        ],
    )

    assert result.exit_code == 2
    assert "all LongBench comparisons failed" in result.stderr
    assert not output_path.exists()
