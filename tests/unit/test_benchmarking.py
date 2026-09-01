"""Tests for deterministic benchmark harnesses."""

from pathlib import Path

from contextos.benchmarking import run_deduplication_benchmark, run_quick_baseline_benchmark
from contextos.tokenization import TiktokenTokenizer


def test_quick_benchmark_is_deterministic() -> None:
    tokenizer = TiktokenTokenizer()
    first = run_quick_baseline_benchmark(tokenizer)
    second = run_quick_baseline_benchmark(tokenizer)
    assert first == second
    assert first["profile"] == "quick"
    assert [result["strategy"] for result in first["results"]] == [
        "full_context",
        "last_n",
        "sliding_window",
        "relevance_only",
        "naive_extractive",
    ]
    assert first["results"][0]["status"] == "overflow"


def test_deduplication_benchmark_reports_labeled_metrics() -> None:
    metrics = run_deduplication_benchmark(Path("benchmarks/datasets/deduplication_cases.json"))
    assert metrics.case_count == 10
    assert metrics.false_positive == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
