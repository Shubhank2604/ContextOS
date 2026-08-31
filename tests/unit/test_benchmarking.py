"""Tests for the deterministic quick benchmark harness."""

from contextos.benchmarking import run_quick_baseline_benchmark
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
    ]
    assert first["results"][0]["status"] == "overflow"
