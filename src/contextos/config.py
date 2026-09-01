"""Configuration models and stable defaults."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contextos.errors import InvalidOptimizationPolicy
from contextos.models import ContextType

DEFAULT_TIKTOKEN_ENCODING = "cl100k_base"
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_TYPE_PRIORITIES: dict[ContextType, float] = {
    ContextType.SYSTEM_INSTRUCTION: 1.0,
    ContextType.TOOL_DEFINITION: 0.5,
    ContextType.USER_MESSAGE: 0.7,
    ContextType.ASSISTANT_MESSAGE: 0.5,
    ContextType.TOOL_OUTPUT: 0.6,
    ContextType.RETRIEVED_DOCUMENT: 0.65,
    ContextType.MEMORY: 0.6,
    ContextType.DECISION: 0.85,
    ContextType.ERROR: 0.8,
    ContextType.PLAN: 0.55,
    ContextType.CODE: 0.75,
    ContextType.TASK_STATE: 0.95,
}


class OptimizationPolicy(BaseModel):
    """Serializable token-budget policy used by every context strategy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_input_tokens: int
    reserve_output_tokens: int = 0
    semantic_dedup_threshold: float = 0.92
    semantic_dedup_enabled: bool = True
    semantic_relevance_enabled: bool = True

    weight_relevance: float = 0.25
    weight_importance: float = 0.20
    weight_recency: float = 0.15
    weight_novelty: float = 0.15
    weight_dependency: float = 0.15
    weight_type_priority: float = 0.10

    position_aware_layout: bool = True
    compression_enabled: bool = True
    compression_target_ratio: float = 0.50
    minimum_compressed_tokens: int = 64

    class_minimum_tokens: dict[ContextType, int] = Field(default_factory=dict)
    class_maximum_tokens: dict[ContextType, int | None] = Field(default_factory=dict)
    recency_half_life_seconds: float = 86_400.0
    dependency_max_depth: int = 2
    type_priorities: dict[ContextType, float] = Field(
        default_factory=lambda: DEFAULT_TYPE_PRIORITIES.copy()
    )

    @model_validator(mode="before")
    @classmethod
    def reject_non_integer_budget_types(cls, data: object) -> object:
        """Reject unsafe coercions with the policy's typed error."""
        if isinstance(data, Mapping):
            max_input_tokens = data.get("max_input_tokens")
            reserve_output_tokens = data.get("reserve_output_tokens", 0)
            if type(max_input_tokens) is not int:
                raise InvalidOptimizationPolicy("max_input_tokens must be a positive integer")
            if type(reserve_output_tokens) is not int:
                raise InvalidOptimizationPolicy(
                    "reserve_output_tokens must be a non-negative integer"
                )
            dependency_max_depth = data.get("dependency_max_depth", 2)
            if type(dependency_max_depth) is not int:
                raise InvalidOptimizationPolicy(
                    "dependency_max_depth must be a non-negative integer"
                )
            minimum_compressed_tokens = data.get("minimum_compressed_tokens", 64)
            if type(minimum_compressed_tokens) is not int:
                raise InvalidOptimizationPolicy(
                    "minimum_compressed_tokens must be a positive integer"
                )
            for field_name, allow_none in (
                ("class_minimum_tokens", False),
                ("class_maximum_tokens", True),
            ):
                configured = data.get(field_name, {})
                if isinstance(configured, Mapping):
                    for value in configured.values():
                        if value is None and allow_none:
                            continue
                        if type(value) is not int:
                            raise InvalidOptimizationPolicy(
                                f"{field_name} values must be non-negative integers"
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
        if not math.isfinite(self.compression_target_ratio) or not (
            0.0 < self.compression_target_ratio < 1.0
        ):
            raise InvalidOptimizationPolicy("compression_target_ratio must be between 0 and 1")
        if self.minimum_compressed_tokens <= 0:
            raise InvalidOptimizationPolicy("minimum_compressed_tokens must be positive")
        weights = self.raw_weights
        if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
            raise InvalidOptimizationPolicy("optimization weights must be non-negative")
        if sum(weights.values()) <= 0:
            raise InvalidOptimizationPolicy("optimization weights must have a positive total")
        if not math.isfinite(self.recency_half_life_seconds) or self.recency_half_life_seconds <= 0:
            raise InvalidOptimizationPolicy("recency_half_life_seconds must be positive")
        if type(self.dependency_max_depth) is not int or self.dependency_max_depth < 0:
            raise InvalidOptimizationPolicy("dependency_max_depth must be a non-negative integer")
        for context_type, minimum in self.class_minimum_tokens.items():
            if type(minimum) is not int or minimum < 0:
                raise InvalidOptimizationPolicy(
                    f"class minimum for {context_type.value} must be a non-negative integer"
                )
            maximum = self.class_maximum_tokens.get(context_type)
            if maximum is not None and minimum > maximum:
                raise InvalidOptimizationPolicy(
                    f"class minimum for {context_type.value} must not exceed its maximum"
                )
        for context_type, maximum in self.class_maximum_tokens.items():
            if maximum is not None and (type(maximum) is not int or maximum < 0):
                raise InvalidOptimizationPolicy(
                    f"class maximum for {context_type.value} must be non-negative or None"
                )
        if any(not 0.0 <= priority <= 1.0 for priority in self.type_priorities.values()):
            raise InvalidOptimizationPolicy("type priorities must be between 0 and 1")

    @property
    def effective_budget(self) -> int:
        """Return the input budget remaining after output reservation."""
        self.validate_static()
        return self.max_input_tokens - self.reserve_output_tokens

    @property
    def raw_weights(self) -> dict[str, float]:
        """Return configured component weights before normalization."""
        return {
            "relevance": self.weight_relevance,
            "importance": self.weight_importance,
            "recency": self.weight_recency,
            "novelty": self.weight_novelty,
            "dependency": self.weight_dependency,
            "type_priority": self.weight_type_priority,
        }

    @property
    def normalized_weights(self) -> dict[str, float]:
        """Return component weights normalized to a sum of one."""
        self.validate_static()
        total = sum(self.raw_weights.values())
        return {name: weight / total for name, weight in self.raw_weights.items()}

    @classmethod
    def quality(
        cls,
        *,
        max_input_tokens: int,
        reserve_output_tokens: int = 0,
        **overrides: object,
    ) -> Self:
        """Favor relevance, importance, and dependency preservation."""
        values: dict[str, object] = {
            "max_input_tokens": max_input_tokens,
            "reserve_output_tokens": reserve_output_tokens,
            "weight_relevance": 0.30,
            "weight_importance": 0.25,
            "weight_recency": 0.10,
            "weight_novelty": 0.10,
            "weight_dependency": 0.20,
            "weight_type_priority": 0.05,
        }
        values.update(overrides)
        return cls.model_validate(values)

    @classmethod
    def balanced(
        cls,
        *,
        max_input_tokens: int,
        reserve_output_tokens: int = 0,
        **overrides: object,
    ) -> Self:
        """Balance all deterministic scoring components."""
        values: dict[str, object] = {
            "max_input_tokens": max_input_tokens,
            "reserve_output_tokens": reserve_output_tokens,
        }
        values.update(overrides)
        return cls.model_validate(values)

    @classmethod
    def economy(
        cls,
        *,
        max_input_tokens: int,
        reserve_output_tokens: int = 0,
        **overrides: object,
    ) -> Self:
        """Favor task relevance, recency, and novelty under tighter selection."""
        values: dict[str, object] = {
            "max_input_tokens": max_input_tokens,
            "reserve_output_tokens": reserve_output_tokens,
            "weight_relevance": 0.30,
            "weight_importance": 0.15,
            "weight_recency": 0.20,
            "weight_novelty": 0.20,
            "weight_dependency": 0.10,
            "weight_type_priority": 0.05,
        }
        values.update(overrides)
        return cls.model_validate(values)
