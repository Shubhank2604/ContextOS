"""Integrity checks for the frozen v0.4 Phase 5A failure taxonomy."""

from __future__ import annotations

import json
from pathlib import Path

TAXONOMY_PATH = Path("benchmarks/analysis/v040_failure_taxonomy.jsonl")
BASELINE_SHA = "4fdd88391300c56ad17af5458897ccdd08d6f7bf"
ALLOWED_FAILURE_CLASSES = {
    "critical_item_under_ranked",
    "class_budget_starvation",
    "dependency_not_preserved",
    "stale_state_retained",
    "contradictory_state_retained",
    "compression_information_loss",
    "incorrect_deduplication",
    "recency_failure",
    "position_failure",
    "type_priority_failure",
    "evaluator_limitation",
    "unknown",
}


def test_v040_failure_taxonomy_is_complete_and_well_formed() -> None:
    records = [
        json.loads(line)
        for line in TAXONOMY_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(records) == 32
    assert len({record["case_id"] for record in records}) == len(records)
    assert {record["family"] for record in records} == {
        "research_agent",
        "support_operations",
    }
    assert all(record["baseline_v040_sha"] == BASELINE_SHA for record in records)
    assert all(record["failure_class"] in ALLOWED_FAILURE_CLASSES for record in records)
    assert all(record["missing_required_items"] for record in records)
    assert all(record["selected_items"] for record in records)
    assert all(record["root_cause"] for record in records)
    assert all(record["hard_constraints_plausibly_address"] for record in records)

    research = [record for record in records if record["family"] == "research_agent"]
    support = [record for record in records if record["family"] == "support_operations"]
    assert len(research) == 16
    assert len(support) == 16
    assert {record["contextos_cir"] for record in research} == {0.0}
    assert {record["contextos_cir"] for record in support} == {0.5}
