"""Tests for optimization policy validation."""

import pytest

from contextos.config import OptimizationPolicy
from contextos.errors import InvalidOptimizationPolicy


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
