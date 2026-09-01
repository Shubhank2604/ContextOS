"""Versioned ContextOS-Bench cases, measurements, and reports."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contextos.config import OptimizationPolicy
from contextos.models import ContextEdge, ContextItem


class BenchmarkFamily(StrEnum):
    """Project-owned benchmark scenario families."""

    CODING_AGENT = "coding_agent"
    RESEARCH_AGENT = "research_agent"
    SUPPORT_OPERATIONS = "support_operations"


class CaseOrigin(StrEnum):
    """Whether a case is a base case or a derived variant."""

    BASE = "base"
    GENERATED_VARIANT = "generated_variant"


class CaseConstruction(StrEnum):
    """How a benchmark case was authored."""

    HAND_AUTHORED = "hand_authored"
    TEMPLATED = "templated"
    GENERATED = "generated"


class TaskMetric(StrEnum):
    """Deterministic task-level scoring methods supported in Phase 4A."""

    REQUIRED_FACT_RECALL = "required_fact_recall"


class RequiredFact(BaseModel):
    """Exact annotated fact that must survive for one source item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str
    value: str
    label: str


class BenchmarkAnswerKey(BaseModel):
    """Typed deterministic evaluation key for a benchmark case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    required_facts: list[RequiredFact]
    expected_answer: str | None = None


class ContextOSBenchCase(BaseModel):
    """One portable, labeled ContextOS-Bench case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.1"
    id: str
    family: BenchmarkFamily
    origin: CaseOrigin
    construction: CaseConstruction
    base_case_id: str | None = None
    generation_seed: int | None = Field(default=None, ge=0)
    task: str
    task_metric: TaskMetric = TaskMetric.REQUIRED_FACT_RECALL
    context_items: list[ContextItem]
    edges: list[ContextEdge] = Field(default_factory=list)
    required_item_ids: list[str]
    answer_key: BenchmarkAnswerKey
    tags: list[str]
    policy: OptimizationPolicy

    @model_validator(mode="after")
    def validate_labels(self) -> ContextOSBenchCase:
        item_ids = [item.id for item in self.context_items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("benchmark context item IDs must be unique")
        required_ids = set(self.required_item_ids)
        unknown = sorted(required_ids - set(item_ids))
        if unknown:
            raise ValueError(f"required item IDs are unknown: {', '.join(unknown)}")
        if not self.required_item_ids or len(self.required_item_ids) != len(required_ids):
            raise ValueError("required item IDs must be non-empty and unique")
        edge_endpoints = {
            endpoint for edge in self.edges for endpoint in (edge.source_id, edge.target_id)
        }
        unknown_edges = sorted(edge_endpoints - set(item_ids))
        if unknown_edges:
            raise ValueError(f"edge endpoint IDs are unknown: {', '.join(unknown_edges)}")
        facts = self.answer_key.required_facts
        if not facts:
            raise ValueError("answer key must contain at least one required fact")
        by_id = {item.id: item for item in self.context_items}
        for fact in facts:
            if fact.item_id not in required_ids:
                raise ValueError(f"fact item {fact.item_id!r} is not a required item")
            if fact.value not in by_id[fact.item_id].content:
                raise ValueError(
                    f"fact {fact.label!r} is not present exactly in item {fact.item_id!r}"
                )
        fact_item_ids = {fact.item_id for fact in facts}
        missing_fact_annotations = sorted(required_ids - fact_item_ids)
        if missing_fact_annotations:
            raise ValueError(
                "required items lack fact annotations: " + ", ".join(missing_fact_annotations)
            )
        if self.origin is CaseOrigin.BASE:
            if self.base_case_id is not None or self.generation_seed is not None:
                raise ValueError("base cases cannot reference generation metadata")
            if self.construction is CaseConstruction.GENERATED:
                raise ValueError("base cases cannot be labeled as generated")
        else:
            if self.base_case_id is None or self.generation_seed is None:
                raise ValueError("generated variants require a base case ID and seed")
            if self.construction is not CaseConstruction.GENERATED:
                raise ValueError("generated variants must use generated construction")
        return self


class ContextOSBenchDataset(BaseModel):
    """A reproducible collection of base cases and generated variants."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.1"
    name: str
    generator_version: str
    generation_seed: int = Field(ge=0)
    cases: list[ContextOSBenchCase]

    @model_validator(mode="after")
    def validate_dataset(self) -> ContextOSBenchDataset:
        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("benchmark case IDs must be unique")
        base_ids = {case.id for case in self.cases if case.origin is CaseOrigin.BASE}
        for case in self.cases:
            if case.origin is CaseOrigin.GENERATED_VARIANT and case.base_case_id not in base_ids:
                raise ValueError(f"generated case {case.id!r} references an unknown base case")
        return self

    @property
    def base_cases(self) -> list[ContextOSBenchCase]:
        """Return base cases in dataset order."""
        return [case for case in self.cases if case.origin is CaseOrigin.BASE]

    @property
    def generated_variants(self) -> list[ContextOSBenchCase]:
        """Return generated variants in dataset order."""
        return [case for case in self.cases if case.origin is CaseOrigin.GENERATED_VARIANT]


class BenchmarkMeasurement(BaseModel):
    """Raw per-case strategy output retained for statistical analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    family: BenchmarkFamily
    strategy: str
    status: str
    task_metric: TaskMetric
    task_specific_score: float = Field(ge=0.0, le=1.0)
    quality_retention: float | None = Field(default=None, ge=0.0)
    critical_information_recall: float = Field(ge=0.0, le=1.0)
    original_tokens: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    effective_budget: int = Field(gt=0)
    context_reduction: float = Field(ge=0.0, le=1.0)
    compression_ratio: float | None = Field(default=None, ge=0.0)
    selected_item_ids: list[str]
    selected_items: list[ContextItem]
    required_items_retained: int = Field(ge=0)
    required_items_total: int = Field(ge=0)
    required_facts_retained: int = Field(ge=0)
    required_facts_total: int = Field(ge=0)
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
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    decision_reasons: dict[str, str] = Field(default_factory=dict)


class ConfidenceInterval(BaseModel):
    """Deterministic percentile-bootstrap confidence interval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    low: float
    high: float
    confidence_level: float = Field(gt=0.0, lt=1.0)
    resamples: int = Field(gt=0)
    seed: int = Field(ge=0)


class BenchmarkAggregate(BaseModel):
    """Strategy-level aggregate metrics calculated from raw cases."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: str
    case_count: int = Field(ge=0)
    successful_case_count: int = Field(ge=0)
    mean_task_specific_score: float = Field(ge=0.0, le=1.0)
    mean_quality_retention: float | None = Field(default=None, ge=0.0)
    mean_critical_information_recall: float = Field(ge=0.0, le=1.0)
    mean_input_tokens: float = Field(ge=0.0)
    mean_context_reduction: float = Field(ge=0.0, le=1.0)
    mean_compression_ratio: float | None = Field(default=None, ge=0.0)
    p50_optimizer_latency_ms: float = Field(ge=0.0)
    p95_optimizer_latency_ms: float = Field(ge=0.0)
    task_score_ci95: ConfidenceInterval | None = None
    cir_ci95: ConfidenceInterval | None = None


class BenchmarkRun(BaseModel):
    """Complete immutable benchmark artifact payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.1"
    run_id: str
    recorded_at_utc: datetime
    dataset_name: str
    dataset_sha256: str
    package_version: str
    python_version: str
    platform: str
    strategies: list[str]
    measurements: list[BenchmarkMeasurement]
    aggregates: list[BenchmarkAggregate]
    metadata: dict[str, Any] = Field(default_factory=dict)
