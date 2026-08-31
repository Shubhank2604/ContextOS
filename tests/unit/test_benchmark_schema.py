"""ContextOS-Bench schema and seed fixture tests."""

import json
from pathlib import Path

from pydantic import TypeAdapter

from contextos.benchmarks import BenchmarkFamily, ContextOSBenchCase


def test_seed_fixture_covers_initial_labeled_families() -> None:
    payload = json.loads(
        Path("benchmarks/datasets/contextos_bench_seed.json").read_text(encoding="utf-8")
    )
    cases = TypeAdapter(list[ContextOSBenchCase]).validate_python(payload)

    assert {case.family for case in cases} == set(BenchmarkFamily)
    assert all(case.required_item_ids for case in cases)
