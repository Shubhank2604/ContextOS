"""Configuration models and stable defaults."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, model_validator

from contextos.errors import InvalidOptimizationPolicy

DEFAULT_TIKTOKEN_ENCODING = "cl100k_base"
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class OptimizationPolicy(BaseModel):
    """Serializable token-budget policy used by every context strategy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_input_tokens: int
    reserve_output_tokens: int = 0
    semantic_dedup_threshold: float = 0.92
    semantic_relevance_enabled: bool = True

    @model_validator(mode="before")
    @classmethod
    def reject_non_integer_budget_types(cls, data: object) -> object:
        """Reject booleans and other coercible values with the policy's typed error."""
        if isinstance(data, Mapping):
            max_input_tokens = data.get("max_input_tokens")
            reserve_output_tokens = data.get("reserve_output_tokens", 0)
            if type(max_input_tokens) is not int:
                raise InvalidOptimizationPolicy("max_input_tokens must be a positive integer")
            if type(reserve_output_tokens) is not int:
                raise InvalidOptimizationPolicy(
                    "reserve_output_tokens must be a non-negative integer"
                )
        return data

    def validate_static(self) -> None:
        """Validate policy-only constraints before tokenization or selection."""
        if type(self.max_input_tokens) is not int or self.max_input_tokens <= 0:
            raise InvalidOptimizationPolicy("max_input_tokens must be a positive integer")
        if type(self.reserve_output_tokens) is not int or self.reserve_output_tokens < 0:
            raise InvalidOptimizationPolicy("reserve_output_tokens must be a non-negative integer")
        if self.reserve_output_tokens >= self.max_input_tokens:
            raise InvalidOptimizationPolicy(
                "max_input_tokens - reserve_output_tokens must be positive"
            )
        if not 0.0 <= self.semantic_dedup_threshold <= 1.0:
            raise InvalidOptimizationPolicy("semantic_dedup_threshold must be between 0 and 1")

    @property
    def effective_budget(self) -> int:
        """Return the input budget remaining after output reservation."""
        self.validate_static()
        return self.max_input_tokens - self.reserve_output_tokens
