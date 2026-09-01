"""Tests for optimization policy validation."""

import pytest

from contextos.config import OptimizationPolicy
from contextos.errors import InvalidOptimizationPolicy
from contextos.models import ContextType


@pytest.mark.parametrize(
    ("max_input_tokens", "reserve_output_tokens", "message"),
    [
        (0, 0, "positive integer"),
        (-1, 0, "positive integer"),
        (10, -1, "non-negative integer"),
        (10, 10, "must be positive"),
        (10, 11, "must be positive"),
        (True, 0, "positive integer"),
        (10, False, "non-negative integer"),
    ],
)
def test_invalid_policy_fails_with_typed_error(
    max_input_tokens: int, reserve_output_tokens: int, message: str
) -> None:
    with pytest.raises(InvalidOptimizationPolicy, match=message):
        policy = OptimizationPolicy(
            max_input_tokens=max_input_tokens,
            reserve_output_tokens=reserve_output_tokens,
        )
        policy.validate_static()


def test_effective_budget_reserves_output_tokens() -> None:
    policy = OptimizationPolicy(max_input_tokens=100, reserve_output_tokens=25)
    assert policy.effective_budget == 75
    assert policy.semantic_dedup_threshold == 0.92
    assert policy.semantic_dedup_enabled is True
    assert policy.semantic_relevance_enabled is True


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_semantic_threshold_must_be_normalized(threshold: float) -> None:
    policy = OptimizationPolicy(max_input_tokens=100, semantic_dedup_threshold=threshold)
    with pytest.raises(InvalidOptimizationPolicy, match="between 0 and 1"):
        policy.validate_static()


def test_policy_contains_complete_runtime_configuration_and_serializes() -> None:
    policy = OptimizationPolicy.balanced(
        max_input_tokens=1_000,
        reserve_output_tokens=100,
        class_minimum_tokens={ContextType.CODE: 20},
        class_maximum_tokens={ContextType.CODE: 200},
    )
    policy.validate_static()
    assert policy.effective_budget == 900
    assert sum(policy.normalized_weights.values()) == pytest.approx(1.0)
    assert policy.position_aware_layout is True
    assert policy.compression_enabled is True
    assert policy.recency_half_life_seconds == 86_400.0
    assert policy.dependency_max_depth == 2
    assert OptimizationPolicy.model_validate_json(policy.model_dump_json()) == policy


@pytest.mark.parametrize(
    "changes",
    [
        {"weight_relevance": -0.1},
        {"weight_relevance": float("nan")},
        {"weight_relevance": float("inf")},
        {"recency_half_life_seconds": 0.0},
        {"recency_half_life_seconds": float("nan")},
        {"dependency_max_depth": -1},
        {"compression_target_ratio": 0.0},
        {"compression_target_ratio": 1.0},
        {"compression_target_ratio": float("nan")},
        {"minimum_compressed_tokens": 0},
        {"class_minimum_tokens": {ContextType.CODE: -1}},
        {"class_maximum_tokens": {ContextType.CODE: -1}},
        {
            "class_minimum_tokens": {ContextType.CODE: 20},
            "class_maximum_tokens": {ContextType.CODE: 10},
        },
        {"type_priorities": {ContextType.CODE: 1.1}},
    ],
)
def test_invalid_runtime_policy_configuration_fails_statically(
    changes: dict[str, object],
) -> None:
    policy = OptimizationPolicy.model_validate({"max_input_tokens": 100, **changes})
    with pytest.raises(InvalidOptimizationPolicy):
        policy.validate_static()


def test_zero_total_weights_fail_statically() -> None:
    policy = OptimizationPolicy(
        max_input_tokens=100,
        weight_relevance=0,
        weight_importance=0,
        weight_recency=0,
        weight_novelty=0,
        weight_dependency=0,
        weight_type_priority=0,
    )
    with pytest.raises(InvalidOptimizationPolicy, match="positive total"):
        policy.validate_static()


def test_profiles_are_deterministic_presets_and_allow_overrides() -> None:
    quality = OptimizationPolicy.quality(max_input_tokens=100)
    balanced = OptimizationPolicy.balanced(max_input_tokens=100)
    economy = OptimizationPolicy.economy(max_input_tokens=100, compression_enabled=False)
    assert quality == OptimizationPolicy.quality(max_input_tokens=100)
    assert quality.raw_weights != balanced.raw_weights != economy.raw_weights
    assert economy.compression_enabled is False
    assert all(
        sum(policy.normalized_weights.values()) == pytest.approx(1.0)
        for policy in (quality, balanced, economy)
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"dependency_max_depth": True},
        {"class_minimum_tokens": {ContextType.CODE: True}},
        {"class_maximum_tokens": {ContextType.CODE: 1.5}},
        {"minimum_compressed_tokens": True},
    ],
)
def test_policy_rejects_coercible_integer_configuration(changes: dict[str, object]) -> None:
    with pytest.raises(InvalidOptimizationPolicy):
        OptimizationPolicy.model_validate({"max_input_tokens": 100, **changes})
