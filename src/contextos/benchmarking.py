"""Small deterministic benchmark harness used by Milestone 1 and CI."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from contextos.baselines import (
    BaselineStrategy,
    FullContextBaseline,
    LastNTokensBaseline,
    SlidingWindowBaseline,
)
from contextos.config import OptimizationPolicy
from contextos.errors import ContextBudgetOverflow
from contextos.models import ContextItem, ContextType
from contextos.tokenization import Tokenizer


def _fixture_items() -> list[ContextItem]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        ContextItem(
            id="old-decision",
            content="Use stateless authentication.",
            type=ContextType.DECISION,
            created_at=start,
            updated_at=start,
            importance=0.9,
        ),
        ContextItem(
            id="recent-noise",
            content="The formatting check passed.",
            type=ContextType.TOOL_OUTPUT,
            created_at=start + timedelta(hours=2),
            updated_at=start + timedelta(hours=2),
            importance=0.2,
        ),
        ContextItem(
            id="current-task",
            content="Fix the authentication timeout.",
            type=ContextType.USER_MESSAGE,
            created_at=start + timedelta(hours=3),
            updated_at=start + timedelta(hours=3),
            importance=1.0,
        ),
    ]


def run_quick_baseline_benchmark(tokenizer: Tokenizer) -> dict[str, Any]:
    """Run the same deterministic fixture through every Milestone 1 baseline."""
    items = _fixture_items()
    total_tokens = sum(tokenizer.count_tokens(item.content) for item in items)
    policy = OptimizationPolicy(max_input_tokens=max(total_tokens - 1, 1), reserve_output_tokens=0)
    strategies: list[BaselineStrategy] = [
        FullContextBaseline(),
        LastNTokensBaseline(),
        SlidingWindowBaseline(window_seconds=2 * 60 * 60),
    ]
    results: list[dict[str, Any]] = []
    for strategy in strategies:
        try:
            result = strategy.optimize(
                task="Fix the authentication timeout.",
                items=items,
                policy=policy,
                tokenizer=tokenizer,
            )
            results.append(
                {
                    "strategy": strategy.name,
                    "status": "ok",
                    "original_tokens": result.original_token_count,
                    "final_tokens": result.final_token_count,
                    "selected_item_ids": [item.id for item in result.selected_items],
                }
            )
        except ContextBudgetOverflow as exc:
            results.append(
                {
                    "strategy": strategy.name,
                    "status": "overflow",
                    "required_tokens": exc.required_tokens,
                    "effective_budget": exc.effective_budget,
                }
            )
    return {"profile": "quick", "case_count": 1, "results": results}
