"""Typed errors shared by ContextOS components."""


class ContextOSError(Exception):
    """Base exception for ContextOS failures."""


class TokenizerError(ContextOSError):
    """Raised when text cannot be tokenized safely."""


class EmbeddingProviderError(ContextOSError):
    """Raised when an embedding provider is unavailable or fails."""


class LLMProviderError(ContextOSError):
    """Raised when an optional language-model provider is unavailable or fails."""


class InvalidEmbeddingOutput(EmbeddingProviderError):
    """Raised when a provider returns malformed or unsafe embeddings."""


class UnknownDependencyReference(ContextOSError):
    """Raised when an edge references an item outside the validated collection."""


class InvalidScore(ContextOSError):
    """Raised when a scoring component is missing or outside its contract."""


class DuplicateContextItemError(ContextOSError):
    """Raised when a collection contains duplicate context item IDs."""


class ContextItemNotFoundError(ContextOSError):
    """Raised when a requested context item is not present in a store."""


class CorruptedStoreError(ContextOSError):
    """Raised when persisted data cannot be opened or decoded safely."""


class StoreMigrationError(ContextOSError):
    """Raised when a store schema cannot be migrated by this runtime."""


class LifecycleError(ContextOSError):
    """Raised when lifecycle instructions or timestamps are invalid."""


class InvalidOptimizationPolicy(ContextOSError):
    """Raised before work begins when a token-budget policy is invalid."""


class MandatoryContextOverflow(ContextOSError):
    """Raised when mandatory context alone exceeds the effective budget."""

    def __init__(self, *, mandatory_tokens: int, effective_budget: int) -> None:
        self.mandatory_tokens = mandatory_tokens
        self.effective_budget = effective_budget
        super().__init__(
            f"mandatory context requires {mandatory_tokens} tokens but the effective budget is "
            f"{effective_budget}"
        )


class ContextualBudgetInfeasible(ContextOSError):
    """Raised when applicable optional class minima exceed optional budget."""

    def __init__(self, *, applicable_minima: int, optional_budget: int) -> None:
        self.applicable_minima = applicable_minima
        self.optional_budget = optional_budget
        super().__init__(
            f"applicable class minima require {applicable_minima} tokens but the optional budget "
            f"is {optional_budget}"
        )


class AllocationError(ContextOSError):
    """Raised when tokenized allocation inputs violate the allocation contract."""


class CompressionError(ContextOSError):
    """Raised when a compressor violates the compression contract."""


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
