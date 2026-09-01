"""ContextOS-Bench schema and canonical dataset tests."""

from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from contextos.benchmarks import BenchmarkFamily, CaseConstruction, CaseOrigin
from contextos.benchmarks.artifacts import load_dataset
from contextos.benchmarks.dataset import build_contextos_bench_dataset
from contextos.benchmarks.models import ContextOSBenchCase

DATASET_PATH = Path("benchmarks/datasets/contextos_bench.json")


def test_canonical_dataset_is_reproducible_and_has_fifty_base_cases() -> None:
    loaded = load_dataset(DATASET_PATH)
    generated = build_contextos_bench_dataset()

    assert loaded == generated
    assert len(loaded.cases) == 50
    assert len(loaded.base_cases) == 50
    assert loaded.generated_variants == []
    assert all(case.origin is CaseOrigin.BASE for case in loaded.cases)
    assert all(case.construction is CaseConstruction.TEMPLATED for case in loaded.cases)


def test_dataset_covers_all_required_scenario_families() -> None:
    dataset = load_dataset(DATASET_PATH)
    counts = Counter(case.family for case in dataset.cases)

    assert counts == {
        BenchmarkFamily.CODING_AGENT: 18,
        BenchmarkFamily.RESEARCH_AGENT: 16,
        BenchmarkFamily.SUPPORT_OPERATIONS: 16,
    }


def test_every_case_contains_phase4a_adversarial_annotations() -> None:
    dataset = load_dataset(DATASET_PATH)
    required_tags = {
        "old_critical_fact",
        "recent_irrelevant_fact",
        "duplicate_paraphrase",
        "changed_number_trap",
        "negation_trap",
        "superseded_decision",
        "contradiction",
        "one_hop_dependency",
        "two_hop_dependency",
        "exact_values",
    }

    for case in dataset.cases:
        assert required_tags <= set(case.tags)
        assert case.required_item_ids
        assert case.answer_key.required_facts
        assert all(fact.value for fact in case.answer_key.required_facts)
        assert len(case.edges) >= 5
        item_by_id = {item.id: item for item in case.context_items}
        required_times = [item_by_id[item_id].updated_at for item_id in case.required_item_ids]
        recent_noise = next(item for item in case.context_items if item.id.endswith("recent-noise"))
        assert max(required_times) < recent_noise.updated_at
        adjacency: dict[str, set[str]] = {item.id: set() for item in case.context_items}
        for edge in case.edges:
            adjacency[edge.source_id].add(edge.target_id)
            adjacency[edge.target_id].add(edge.source_id)
        task_id = next(item.id for item in case.context_items if item.id.endswith("-task"))
        one_hop = adjacency[task_id]
        two_hop = {
            neighbor
            for item_id in one_hop
            for neighbor in adjacency[item_id]
            if neighbor != task_id and neighbor not in one_hop
        }
        assert one_hop
        assert set(case.required_item_ids) & two_hop


def test_generated_variant_requires_reproducibility_metadata() -> None:
    base = build_contextos_bench_dataset().cases[0]
    invalid = base.model_dump(mode="python")
    invalid.update(
        {
            "id": "generated-without-metadata",
            "origin": CaseOrigin.GENERATED_VARIANT,
            "construction": CaseConstruction.GENERATED,
        }
    )

    with pytest.raises(ValidationError, match="base case ID and seed"):
        ContextOSBenchCase.model_validate(invalid)
