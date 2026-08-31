"""Versioned ContextOS-Bench input and raw measurement schemas."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contextos.config import OptimizationPolicy
from contextos.models import ContextEdge, ContextItem


class BenchmarkFamily(StrEnum):
    """Initial labeled benchmark families."""

    DEDUPLICATION = "deduplication"
    ALLOCATION = "allocation"
    REGRESSION = "regression"


class ContextOSBenchCase(BaseModel):
    """One portable, labeled ContextOS-Bench case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    id: str
    family: BenchmarkFamily
    task: str
    context_items: list[ContextItem]
    edges: list[ContextEdge] = Field(default_factory=list)
    required_item_ids: list[str] = Field(default_factory=list)
    answer_key: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    policy: OptimizationPolicy

    @model_validator(mode="after")
    def validate_labels(self) -> ContextOSBenchCase:
        item_ids = {item.id for item in self.context_items}
        unknown = sorted(set(self.required_item_ids) - item_ids)
        if unknown:
            raise ValueError(f"required item IDs are unknown: {', '.join(unknown)}")
        return self


class BenchmarkMeasurement(BaseModel):
    """Raw strategy output retained for later statistical analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    strategy: str
    status: str
    original_tokens: int = Field(ge=0)
    final_tokens: int = Field(ge=0)
    effective_budget: int = Field(gt=0)
    selected_item_ids: list[str]
    required_items_retained: int = Field(ge=0)
    required_items_total: int = Field(ge=0)
    compressed_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    optimizer_wall_time_ms: float = Field(ge=0.0)
    embedding_time_ms: float = Field(ge=0.0)
    compression_time_ms: float = Field(ge=0.0)
    model_ttft_ms: float | None = Field(default=None, ge=0.0)
    model_total_latency_ms: float | None = Field(default=None, ge=0.0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)
    provider_model: str | None = None
    pricing_config: dict[str, float] | None = None
