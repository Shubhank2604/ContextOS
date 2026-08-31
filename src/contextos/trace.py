"""Explainable per-item and whole-run optimization traces."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from contextos.config import OptimizationPolicy
from contextos.models import ContextItem


class OptimizationDecision(StrEnum):
    """Final disposition of a candidate context item."""

    RETAINED = "retained"
    REMOVED = "removed"
    COMPRESSED = "compressed"


class ItemTrace(BaseModel):
    """All currently available evidence for one item decision."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    initial_token_count: int = Field(ge=0)

    exact_duplicate_of: str | None = None
    semantic_duplicate_of: str | None = None
    semantic_similarity: float | None = Field(default=None, ge=0.0, le=1.0)

    relevance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    importance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    recency_score: float | None = Field(default=None, ge=0.0, le=1.0)
    novelty_score: float | None = Field(default=None, ge=0.0, le=1.0)
    dependency_score: float | None = Field(default=None, ge=0.0, le=1.0)
    type_priority: float | None = Field(default=None, ge=0.0, le=1.0)
    composite_utility: float | None = Field(default=None, ge=0.0, le=1.0)
    value_density: float | None = Field(default=None, ge=0.0)

    decision: OptimizationDecision
    decision_reason: str

    final_token_count: int = Field(ge=0)
    final_position: int | None = Field(default=None, ge=0)
    compression_strategy: str | None = None
    provenance: list[str] = Field(default_factory=list)


class OptimizationTrace(BaseModel):
    """Whole-run budget, timing, warning, and item-decision record."""

    model_config = ConfigDict(extra="forbid")

    strategy: str
    policy: OptimizationPolicy
    effective_budget: int = Field(gt=0)
    mandatory_tokens: int = Field(ge=0)
    optional_budget: int = Field(ge=0)
    original_tokens: int = Field(ge=0)
    final_tokens: int = Field(ge=0)
    reduction_ratio: float = Field(ge=0.0, le=1.0)
    stage_timings_ms: dict[str, float]
    selected_count: int = Field(ge=0)
    removed_count: int = Field(ge=0)
    compressed_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    items: list[ItemTrace]


class BudgetAllocation(BaseModel):
    """Budget use for an optimized context result."""

    model_config = ConfigDict(extra="forbid")

    effective_budget: int = Field(gt=0)
    used_tokens: int = Field(ge=0)
    remaining_tokens: int = Field(ge=0)


class OptimizedContext(BaseModel):
    """Selected context plus removals, budget accounting, and trace."""

    model_config = ConfigDict(extra="forbid")

    selected_items: list[ContextItem]
    removed_items: list[ContextItem]
    original_token_count: int = Field(ge=0)
    final_token_count: int = Field(ge=0)
    budget_allocation: BudgetAllocation
    trace: OptimizationTrace
    metadata: dict[str, Any] = Field(default_factory=dict)
