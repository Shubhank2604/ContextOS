"""End-to-end acceptance tests for the integrated v0.3 optimizer."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from contextos import ContextEdge, ContextItem, ContextOptimizer, ContextType, OptimizationPolicy
from contextos.errors import EmbeddingProviderError, MandatoryContextOverflow
from contextos.models import DependencyRelation
from contextos.store import SQLiteContextStore


class WordTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text.split())


class UnavailableEmbeddingProvider:
    @property
    def identity(self) -> str:
        return "unavailable"

    def embed(self, texts: Sequence[str]) -> NDArray[np.float64]:
        del texts
        raise EmbeddingProviderError("offline")


def make_item(
    item_id: str,
    content: str,
    context_type: ContextType,
    position: int,
    *,
    mandatory: bool = False,
) -> ContextItem:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=position)
    return ContextItem(
        id=item_id,
        content=content,
        type=context_type,
        created_at=timestamp,
        updated_at=timestamp,
        mandatory=mandatory,
        evictable=not mandatory,
    )


def policy(budget: int) -> OptimizationPolicy:
    return OptimizationPolicy(
        max_input_tokens=budget,
        compression_target_ratio=0.4,
        minimum_compressed_tokens=2,
    )


def test_full_mixed_context_pipeline_is_deterministic_and_fully_traced() -> None:
    items = [
        make_item(
            "system",
            "Follow safety rules",
            ContextType.SYSTEM_INSTRUCTION,
            0,
            mandatory=True,
        ),
        make_item(
            "doc",
            "Authentication timeout is 37 seconds. Extra background.",
            ContextType.RETRIEVED_DOCUMENT,
            1,
        ),
        make_item("state", "Current task is timeout repair", ContextType.TASK_STATE, 2),
        make_item("noise", "Formatting completed normally", ContextType.TOOL_OUTPUT, 3),
    ]
    optimizer = ContextOptimizer(tokenizer=WordTokenizer())

    first = optimizer.optimize("repair authentication timeout", items, policy(12))
    second = optimizer.optimize("repair authentication timeout", items, policy(12))

    assert first.final_token_count <= 12
    assert first.selected_items[0].id == "system"
    assert [item.id for item in first.selected_items] == [item.id for item in second.selected_items]
    assert first.trace.items[0].final_position == 0
    assert set(first.trace.stage_timings_ms) == {
        "validate_policy",
        "tokenize",
        "reserve_mandatory",
        "exact_dedup",
        "semantic_dedup",
        "contextual_budget_validation",
        "relevance",
        "importance",
        "recency",
        "novelty",
        "dependencies",
        "composite_utility",
        "allocation_plan",
        "compression",
        "final_selection",
        "layout",
        "invariant_validation",
        "trace",
        "lifecycle_persistence",
    }


def test_sqlite_persistence_is_part_of_integrated_pipeline(tmp_path: Path) -> None:
    store = SQLiteContextStore(tmp_path / "optimizer.db")
    source = make_item("persisted", "durable optimizer input", ContextType.MEMORY, 0)
    ContextOptimizer(tokenizer=WordTokenizer(), store=store).optimize(
        "durability", [source], policy(10)
    )
    store.close()

    reopened = SQLiteContextStore(tmp_path / "optimizer.db")
    assert reopened.load_item("persisted").token_count == 3
    reopened.close()


def test_unavailable_semantic_provider_has_visible_offline_fallback() -> None:
    source = make_item("source", "offline provider context", ContextType.MEMORY, 0)
    result = ContextOptimizer(
        tokenizer=WordTokenizer(), embedding_provider=UnavailableEmbeddingProvider()
    ).optimize("task", [source], policy(10))

    assert "embedding_provider_unavailable:deterministic_fallback" in result.trace.warnings


def test_mandatory_overflow_fails_before_deduplication() -> None:
    mandatory = make_item(
        "mandatory", "one two three four", ContextType.SYSTEM_INSTRUCTION, 0, mandatory=True
    )
    with pytest.raises(MandatoryContextOverflow):
        ContextOptimizer(tokenizer=WordTokenizer()).optimize("task", [mandatory], policy(3))


def test_empty_optional_context_and_cyclic_dependencies_are_supported() -> None:
    mandatory = make_item("system", "required", ContextType.SYSTEM_INSTRUCTION, 0, mandatory=True)
    only_mandatory = ContextOptimizer(tokenizer=WordTokenizer()).optimize(
        "task", [mandatory], policy(5)
    )
    assert [item.id for item in only_mandatory.selected_items] == ["system"]

    first = make_item("first", "first dependency", ContextType.DECISION, 1)
    second = make_item("second", "second dependency", ContextType.PLAN, 2)
    edges = [
        ContextEdge(
            source_id="first",
            target_id="second",
            relation=DependencyRelation.REQUIRES,
            weight=1.0,
        ),
        ContextEdge(
            source_id="second",
            target_id="first",
            relation=DependencyRelation.RELATED_TO,
            weight=0.5,
        ),
    ]
    cyclic = ContextOptimizer(tokenizer=WordTokenizer(), edges=edges).optimize(
        "dependency", [first, second], policy(10)
    )
    assert {item.id for item in cyclic.selected_items} == {"first", "second"}


def test_large_tool_output_compresses_without_losing_error_line() -> None:
    log = make_item(
        "log",
        "start request\nnoise one two three four five\nERROR status=503 request_id=ABC-42\n"
        "more irrelevant output values here\nend request",
        ContextType.TOOL_OUTPUT,
        0,
    )
    result = ContextOptimizer(tokenizer=WordTokenizer()).optimize(
        "request failed", [log], policy(8)
    )

    selected = result.selected_items[0]
    assert selected.metadata["compression_strategy"] == "tool_output"
    assert "ERROR status=503 request_id=ABC-42" in selected.content
    assert result.final_token_count <= 8


def test_repeated_numeric_different_items_are_not_semantically_collapsed() -> None:
    first = make_item("timeout-30", "Timeout is 30 seconds.", ContextType.DECISION, 0)
    second = make_item("timeout-60", "Timeout is 60 seconds.", ContextType.DECISION, 1)
    result = ContextOptimizer(tokenizer=WordTokenizer()).optimize(
        "timeout", [first, second], policy(20)
    )

    assert {item.id for item in result.selected_items} == {"timeout-30", "timeout-60"}
