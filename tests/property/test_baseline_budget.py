"""Property tests for baseline token-budget invariants."""

from datetime import UTC, datetime, timedelta

from hypothesis import given
from hypothesis import strategies as st

from contextos.baselines import BaselineStrategy, LastNTokensBaseline, SlidingWindowBaseline
from contextos.config import OptimizationPolicy
from contextos.models import ContextItem, ContextType


class CharacterTokenizer:
    """Count each character as one deterministic token."""

    def count_tokens(self, text: str) -> int:
        return len(text)


@given(
    contents=st.lists(st.text(alphabet="abc123", min_size=1, max_size=12), min_size=0, max_size=12),
    budget=st.integers(min_value=1, max_value=30),
)
def test_selecting_baselines_never_exceed_budget(contents: list[str], budget: int) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    items = [
        ContextItem(
            id=f"item-{index}",
            content=content,
            type=ContextType.MEMORY,
            created_at=start + timedelta(seconds=index),
            updated_at=start + timedelta(seconds=index),
            importance=0.5,
        )
        for index, content in enumerate(contents)
    ]
    policy = OptimizationPolicy(max_input_tokens=budget)
    strategies: list[BaselineStrategy] = [
        LastNTokensBaseline(),
        SlidingWindowBaseline(window_seconds=60),
    ]
    for strategy in strategies:
        result = strategy.optimize(
            task="property test",
            items=items,
            policy=policy,
            tokenizer=CharacterTokenizer(),
        )
        assert result.final_token_count <= policy.effective_budget
        assert result.final_token_count == sum(
            item.token_count or 0 for item in result.selected_items
        )
