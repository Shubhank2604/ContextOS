"""Optimize a minimal context through the public SDK."""

from datetime import UTC, datetime

from contextos import ContextItem, ContextOptimizer, ContextType, OptimizationPolicy

item = ContextItem(
    id="task-1",
    content="Fix the authentication timeout without changing the stateless design.",
    type=ContextType.USER_MESSAGE,
    created_at=datetime.now(UTC),
    updated_at=datetime.now(UTC),
    importance=1.0,
    mandatory=True,
    evictable=False,
)

result = ContextOptimizer().optimize(
    "Fix the authentication timeout",
    [item],
    OptimizationPolicy.balanced(max_input_tokens=100),
)
print(result.model_dump_json(indent=2))
