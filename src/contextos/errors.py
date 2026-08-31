"""Typed errors shared by ContextOS components."""


class ContextOSError(Exception):
    """Base exception for ContextOS failures."""


class TokenizerError(ContextOSError):
    """Raised when text cannot be tokenized safely."""


class DuplicateContextItemError(ContextOSError):
    """Raised when a collection contains duplicate context item IDs."""


class ContextItemNotFoundError(ContextOSError):
    """Raised when a requested context item is not present in a store."""


class InvalidOptimizationPolicy(ContextOSError):
    """Raised before work begins when a token-budget policy is invalid."""


class ContextBudgetOverflow(ContextOSError):
    """Raised when a strategy cannot represent its result within budget."""

    def __init__(self, *, strategy: str, required_tokens: int, effective_budget: int) -> None:
        self.strategy = strategy
        self.required_tokens = required_tokens
        self.effective_budget = effective_budget
        super().__init__(
            f"{strategy} requires {required_tokens} input tokens but the effective budget is "
            f"{effective_budget}"
        )
