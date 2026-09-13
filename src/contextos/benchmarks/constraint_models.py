"""Schemas for the Phase 5 constraint-sensitive benchmark track."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contextos.benchmarks.models import (
    BenchmarkMeasurement,
    ContextOSBenchCase,
)
from contextos.models import ContextEdge


class ConstraintCategory(StrEnum):
    """Deterministically scored preservation failure families."""

    DEPENDENCY = "dependency"
    SUPERSESSION = "supersession"
    CONTRADICTION = "contradiction"
    EXACT_NUMERIC = "exact_numeric"
    IDENTIFIER = "identifier"
    NEGATION_POLICY = "negation_policy"
    TOOL_STATE = "tool_state"
    CITATION_EVIDENCE = "citation_evidence"
    MULTI_HOP = "multi_hop"


class ConstraintGroundTruth(BaseModel):
    """Machine-readable legal-state requirements for one case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: ConstraintCategory
    critical_item_ids: list[str]
    relations: list[ContextEdge] = Field(default_factory=list)
    required_exact_values: list[str] = Field(default_factory=list)
    required_identifiers: list[str] = Field(default_factory=list)
    required_citations: list[str] = Field(default_factory=list)
    forbidden_item_combinations: list[tuple[str, ...]] = Field(default_factory=list)
    expected_current_state: dict[str, str] = Field(default_factory=dict)
    stale_item_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ground_truth(self) -> ConstraintGroundTruth:
        if not self.critical_item_ids:
            raise ValueError("constraint cases require critical item IDs")
        if any(len(group) < 2 for group in self.forbidden_item_combinations):
            raise ValueError("forbidden item combinations require at least two items")
        return self


class ConstraintBenchmarkCase(ContextOSBenchCase):
    """A ContextOS-Bench-compatible case with preservation ground truth."""

    schema_version: str = "2.0"
    constraints: ConstraintGroundTruth

    @model_validator(mode="after")
    def validate_constraints(self) -> ConstraintBenchmarkCase:
        item_ids = {item.id for item in self.context_items}
        referenced = set(self.constraints.critical_item_ids)
        referenced.update(self.constraints.stale_item_ids)
        referenced.update(
            item_id for group in self.constraints.forbidden_item_combinations for item_id in group
        )
        referenced.update(
            endpoint
            for edge in self.constraints.relations
            for endpoint in (edge.source_id, edge.target_id)
        )
        unknown = sorted(referenced - item_ids)
        if unknown:
            raise ValueError(f"constraint annotations reference unknown items: {unknown}")
        return self


class ConstraintBenchmarkDataset(BaseModel):
    """Separate Phase 5 benchmark track; does not replace ContextOS-Bench."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "2.0"
    name: str
    generator_version: str
    generation_seed: int = Field(ge=0)
    cases: list[ConstraintBenchmarkCase]

    @model_validator(mode="after")
    def validate_dataset(self) -> ConstraintBenchmarkDataset:
        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("constraint benchmark case IDs must be unique")
        return self


class ConstraintMeasurement(BaseModel):
    """Per-case constraint metrics alongside the unchanged v0.4 measurement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark: BenchmarkMeasurement
    category: ConstraintCategory
    constraint_violation_rate: float = Field(ge=0.0, le=1.0)
    dependency_closure_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    state_consistency_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    contradiction_leakage_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    exact_value_preservation_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    identifier_preservation_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_preservation_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class ConstraintAggregate(BaseModel):
    """Strategy-level means for the constraint-specific metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: str
    case_count: int = Field(ge=0)
    constraint_violation_rate: float = Field(ge=0.0, le=1.0)
    dependency_closure_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    state_consistency_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    contradiction_leakage_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    exact_value_preservation_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    identifier_preservation_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_preservation_rate: float | None = Field(default=None, ge=0.0, le=1.0)
