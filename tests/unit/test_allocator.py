"""Tests for contextual validation and deterministic allocation planning."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from contextos.budget import (
    AllocationPlan,
    CompressionRequest,
    DirectSelection,
    TokenBudgetAllocator,
    validate_contextual_budget,
)
from contextos.config import OptimizationPolicy
from contextos.errors import (
    AllocationError,
    ContextualBudgetInfeasible,
    InvalidScore,
    MandatoryContextOverflow,
)
from contextos.models import ContextItem, ContextType
from contextos.scoring import ScoreBreakdown


def _item(
    item_id: str,
    tokens: int | None,
    *,
    context_type: ContextType = ContextType.MEMORY,
    mandatory: bool = False,
    compressible: bool = True,
) -> ContextItem:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return ContextItem(
        id=item_id,
        content=f"content for {item_id}",
        type=context_type,
        created_at=timestamp,
        updated_at=timestamp,
        importance=0.5,
        mandatory=mandatory,
        evictable=not mandatory,
        compressible=compressible,
        token_count=tokens,
    )


def _score(utility: float) -> ScoreBreakdown:
    return ScoreBreakdown(
        relevance=utility,
        importance=utility,
        recency=utility,
        novelty=utility,
        dependency=utility,
        type_priority=utility,
        composite_utility=utility,
    )


def _allocate(
    items: list[ContextItem],
    utilities: dict[str, float],
    policy: OptimizationPolicy,
) -> AllocationPlan:
    return TokenBudgetAllocator().allocate(
        items,
        scores={item_id: _score(utility) for item_id, utility in utilities.items()},
        policy=policy,
    )


def test_contextual_budget_reserves_mandatory_and_ignores_absent_minima() -> None:
    result = validate_contextual_budget(
        [
            _item("mandatory", 30, mandatory=True),
            _item("optional", 20, context_type=ContextType.MEMORY),
        ],
        policy=OptimizationPolicy(
            max_input_tokens=100,
            reserve_output_tokens=10,
            class_minimum_tokens={ContextType.CODE: 80},
        ),
    )
    assert result.effective_budget == 90
    assert result.mandatory_tokens == 30
    assert result.optional_budget == 60
    assert result.applicable_minima == 0
    assert result.present_optional_types == (ContextType.MEMORY,)


def test_mandatory_overflow_and_contextual_minimum_infeasibility_are_distinct() -> None:
    with pytest.raises(MandatoryContextOverflow) as overflow:
        validate_contextual_budget(
            [_item("mandatory", 51, mandatory=True)],
            policy=OptimizationPolicy(max_input_tokens=50),
        )
    assert overflow.value.mandatory_tokens == 51
    assert overflow.value.effective_budget == 50

    with pytest.raises(ContextualBudgetInfeasible) as infeasible:
        validate_contextual_budget(
            [_item("code", 10, context_type=ContextType.CODE)],
            policy=OptimizationPolicy(
                max_input_tokens=50,
                class_minimum_tokens={ContextType.CODE: 60},
            ),
        )
    assert infeasible.value.applicable_minima == 60
    assert infeasible.value.optional_budget == 50


def test_allocation_requires_token_counts_and_optional_scores() -> None:
    with pytest.raises(AllocationError, match="not been tokenized"):
        _allocate(
            [_item("missing-token-count", None)],
            {"missing-token-count": 0.5},
            OptimizationPolicy(max_input_tokens=10),
        )
    with pytest.raises(InvalidScore, match="missing"):
        _allocate(
            [_item("missing-score", 1)],
            {},
            OptimizationPolicy(max_input_tokens=10),
        )


def test_pass_a_uses_stable_type_order_and_allows_floor_granularity() -> None:
    items = [
        _item("code", 7, context_type=ContextType.CODE),
        _item("tool", 5, context_type=ContextType.TOOL_DEFINITION),
    ]
    plan = _allocate(
        items,
        {"code": 0.9, "tool": 0.1},
        OptimizationPolicy(
            max_input_tokens=12,
            compression_enabled=False,
            class_minimum_tokens={ContextType.CODE: 5, ContextType.TOOL_DEFINITION: 5},
            class_maximum_tokens={ContextType.CODE: 7, ContextType.TOOL_DEFINITION: 5},
        ),
    )
    assert [selection.item_id for selection in plan.direct_selected] == ["tool", "code"]
    assert all(selection.reason == "class_soft_floor" for selection in plan.direct_selected)
    assert plan.class_allocated_tokens == {
        ContextType.TOOL_DEFINITION: 5,
        ContextType.CODE: 7,
    }


def test_unavailable_floor_content_is_selected_and_traced() -> None:
    plan = _allocate(
        [_item("only", 5, context_type=ContextType.CODE)],
        {"only": 0.5},
        OptimizationPolicy(
            max_input_tokens=20,
            compression_enabled=False,
            class_minimum_tokens={ContextType.CODE: 10},
        ),
    )
    assert [selection.item_id for selection in plan.direct_selected] == ["only"]
    assert plan.minimum_unmet_reasons == {ContextType.CODE: "no_more_candidates"}


def test_global_pass_ranks_density_then_utility_then_id() -> None:
    items = [_item("z", 10), _item("b", 5), _item("a", 5)]
    plan = _allocate(
        items,
        {"z": 1.0, "b": 0.5, "a": 0.5},
        OptimizationPolicy(max_input_tokens=10, compression_enabled=False),
    )
    assert [selection.item_id for selection in plan.direct_selected] == ["z"]
    assert plan.rejected_item_ids == ["a", "b"]


def test_class_maximum_is_never_exceeded() -> None:
    plan = _allocate(
        [
            _item("a", 5, context_type=ContextType.CODE),
            _item("b", 5, context_type=ContextType.CODE),
        ],
        {"a": 0.8, "b": 0.7},
        OptimizationPolicy(
            max_input_tokens=20,
            compression_enabled=False,
            class_maximum_tokens={ContextType.CODE: 6},
        ),
    )
    assert [selection.item_id for selection in plan.direct_selected] == ["a"]
    assert plan.rejection_reasons == {"b": "class_maximum_exceeded"}
    assert plan.class_allocated_tokens[ContextType.CODE] == 5


def test_compression_planning_uses_planned_density_and_reserves_budget() -> None:
    items = [_item("a", 120), _item("b", 110)]
    plan = _allocate(
        items,
        {"a": 0.8, "b": 0.9},
        OptimizationPolicy(
            max_input_tokens=60,
            compression_target_ratio=0.5,
            minimum_compressed_tokens=10,
        ),
    )
    assert [request.item_id for request in plan.compression_requests] == ["b"]
    assert plan.compression_requests[0].target_tokens == 55
    assert plan.rejection_reasons == {"a": "insufficient_compression_budget"}
    assert plan.compression_budget == 60
    assert plan.reserved_compression_tokens == 55
    assert plan.remaining_compression_tokens == 5
    assert plan.compression_candidate_order == ["b", "a"]
    assert plan.compression_candidate_targets == {"b": 55, "a": 60}


def test_compression_target_rounds_up_and_must_be_beneficial() -> None:
    beneficial = _allocate(
        [_item("beneficial", 101)],
        {"beneficial": 1.0},
        OptimizationPolicy(
            max_input_tokens=60,
            compression_target_ratio=0.5,
            minimum_compressed_tokens=1,
        ),
    )
    assert beneficial.compression_requests[0].target_tokens == 51

    not_beneficial = _allocate(
        [_item("small", 50)],
        {"small": 1.0},
        OptimizationPolicy(max_input_tokens=10, minimum_compressed_tokens=64),
    )
    assert not_beneficial.rejection_reasons == {"small": "compression_not_beneficial"}


def test_compressed_reservation_also_obeys_class_maximum() -> None:
    plan = _allocate(
        [_item("code", 20, context_type=ContextType.CODE)],
        {"code": 1.0},
        OptimizationPolicy(
            max_input_tokens=10,
            compression_target_ratio=0.5,
            minimum_compressed_tokens=1,
            class_maximum_tokens={ContextType.CODE: 5},
        ),
    )
    assert plan.compression_requests == []
    assert plan.rejection_reasons == {"code": "class_maximum_exceeded"}


@pytest.mark.parametrize(
    ("compressible", "compression_enabled", "expected_reason"),
    [
        (False, True, "not_compressible"),
        (True, False, "compression_disabled"),
    ],
)
def test_non_compression_candidates_are_rejected_explicitly(
    compressible: bool,
    compression_enabled: bool,
    expected_reason: str,
) -> None:
    plan = _allocate(
        [_item("item", 20, compressible=compressible)],
        {"item": 1.0},
        OptimizationPolicy(max_input_tokens=10, compression_enabled=compression_enabled),
    )
    assert plan.rejection_reasons == {"item": expected_reason}


def test_allocation_is_deterministic_and_partitions_every_candidate() -> None:
    items = [_item("a", 8), _item("b", 8), _item("c", 8)]
    policy = OptimizationPolicy(
        max_input_tokens=12,
        compression_target_ratio=0.5,
        minimum_compressed_tokens=2,
    )
    first = _allocate(items, {"a": 0.8, "b": 0.7, "c": 0.6}, policy)
    second = _allocate(items, {"a": 0.8, "b": 0.7, "c": 0.6}, policy)
    assert first == second
    outcomes = {
        *(selection.item_id for selection in first.direct_selected),
        *(request.item_id for request in first.compression_requests),
        *first.rejected_item_ids,
    }
    assert outcomes == {"a", "b", "c"}


def test_allocation_plan_rejects_duplicate_outcomes_and_bad_accounting() -> None:
    selection = DirectSelection(
        item_id="item",
        allocated_tokens=5,
        utility=0.5,
        value_density=0.1,
        reason="test",
    )
    with pytest.raises(ValidationError, match="more than one outcome"):
        AllocationPlan(
            direct_selected=[selection],
            compression_requests=[],
            rejected_item_ids=["item"],
            optional_budget=10,
            direct_tokens=5,
            compression_budget=5,
            candidate_item_ids=["item"],
            rejection_reasons={"item": "test"},
        )


def test_compression_request_must_reduce_tokens() -> None:
    with pytest.raises(ValidationError, match="smaller"):
        CompressionRequest(
            item_id="item",
            target_tokens=10,
            original_tokens=10,
            utility=0.5,
            value_density=0.05,
            reason="test",
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"candidate_item_ids": []}, "exactly one outcome"),
        ({"candidate_item_ids": ["item", "item"]}, "unique"),
        ({"rejection_reasons": {"other": "test"}}, "rejection reason"),
        ({"direct_tokens": 4, "compression_budget": 6}, "direct_tokens"),
        ({"compression_budget": 4}, "optional_budget - direct_tokens"),
        (
            {
                "compression_candidate_order": ["candidate", "candidate"],
                "compression_candidate_targets": {"candidate": 2},
            },
            "unique IDs",
        ),
        (
            {
                "compression_candidate_order": ["candidate"],
                "compression_candidate_targets": {},
            },
            "one target",
        ),
    ],
)
def test_allocation_plan_validates_partition_metadata(
    changes: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "direct_selected": [
            DirectSelection(
                item_id="item",
                allocated_tokens=5,
                utility=0.5,
                value_density=0.1,
                reason="test",
            )
        ],
        "compression_requests": [],
        "rejected_item_ids": [],
        "optional_budget": 10,
        "direct_tokens": 5,
        "compression_budget": 5,
        "candidate_item_ids": ["item"],
        "rejection_reasons": {},
    }
    values.update(changes)
    with pytest.raises(ValidationError, match=message):
        AllocationPlan.model_validate(values)


def test_allocation_plan_rejects_excess_reservations_and_invalid_candidate_target() -> None:
    request = CompressionRequest(
        item_id="item",
        target_tokens=6,
        original_tokens=10,
        utility=0.5,
        value_density=0.05,
        reason="test",
    )
    with pytest.raises(ValidationError, match="reservations exceed"):
        AllocationPlan(
            direct_selected=[],
            compression_requests=[request],
            rejected_item_ids=[],
            optional_budget=5,
            direct_tokens=0,
            compression_budget=5,
            candidate_item_ids=["item"],
            rejection_reasons={},
            compression_candidate_order=["item"],
            compression_candidate_targets={"item": 6},
        )

    with pytest.raises(ValidationError, match="targets must be positive"):
        AllocationPlan(
            direct_selected=[],
            compression_requests=[],
            rejected_item_ids=["item"],
            optional_budget=5,
            direct_tokens=0,
            compression_budget=5,
            candidate_item_ids=["item"],
            rejection_reasons={"item": "test"},
            compression_candidate_order=["item"],
            compression_candidate_targets={"item": 0},
        )
