"""Validated schemas for the controlled positional-retrieval experiment."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidencePosition(StrEnum):
    """Requested placement of the relevant record in the original context."""

    BEGINNING = "beginning"
    QUARTER = "25_percent"
    MIDDLE = "middle"
    THREE_QUARTER = "75_percent"
    END = "end"

    @property
    def fraction(self) -> float:
        """Return the requested fractional location in the record sequence."""
        return {
            EvidencePosition.BEGINNING: 0.0,
            EvidencePosition.QUARTER: 0.25,
            EvidencePosition.MIDDLE: 0.5,
            EvidencePosition.THREE_QUARTER: 0.75,
            EvidencePosition.END: 1.0,
        }[self]


class PositionalStrategy(StrEnum):
    """Layouts compared while holding evidence and provider configuration fixed."""

    ORIGINAL_FULL = "original_full"
    RELEVANCE_DESCENDING = "relevance_descending"
    CONTEXTOS_POSITION_AWARE = "contextos_position_aware"


class PositionalCase(BaseModel):
    """Compact deterministic recipe for one controlled retrieval prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    id: str
    target_context_tokens: int = Field(ge=256)
    evidence_position: EvidencePosition
    repetition: int = Field(ge=0)
    seed: int = Field(ge=0)
    target_key: str
    expected_value: str


class PositionalDataset(BaseModel):
    """Reproducible parameter grid for the controlled experiment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    name: str = "contextos-positional-retrieval"
    generator_version: str
    generation_seed: int = Field(ge=0)
    cases: list[PositionalCase]

    @model_validator(mode="after")
    def validate_unique_cases(self) -> PositionalDataset:
        case_ids = [case.id for case in self.cases]
        if not self.cases or len(case_ids) != len(set(case_ids)):
            raise ValueError("positional dataset case IDs must be non-empty and unique")
        return self


class PositionalPrediction(BaseModel):
    """Raw exact-match result for one case, strategy, and provider invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    strategy: PositionalStrategy
    target_context_tokens: int = Field(ge=256)
    evidence_position: EvidencePosition
    actual_evidence_fraction: float = Field(ge=0.0, le=1.0)
    distractor_count: int = Field(ge=1)
    expected_value: str
    raw_prediction: str
    normalized_prediction: str
    exact_match: bool
    estimated_input_tokens: int = Field(ge=0)
    provider_input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    model_ttft_ms: float | None = Field(default=None, ge=0.0)
    model_total_latency_ms: float = Field(ge=0.0)
    prompt_sha256: str


class PositionAccuracy(BaseModel):
    """Accuracy and systems measurements for one aggregation cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: PositionalStrategy
    target_context_tokens: int = Field(ge=256)
    evidence_position: EvidencePosition
    case_count: int = Field(gt=0)
    correct_count: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    mean_input_tokens: float = Field(ge=0.0)
    mean_model_latency_ms: float = Field(ge=0.0)


class PositionalRobustness(BaseModel):
    """Position sensitivity for one strategy and context-length bucket."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: PositionalStrategy
    target_context_tokens: int = Field(ge=256)
    accuracy_by_position: dict[EvidencePosition, float]
    mean_accuracy: float = Field(ge=0.0, le=1.0)
    max_min_positional_gap: float = Field(ge=0.0, le=1.0)
    positional_variance: float = Field(ge=0.0)
    positional_std_dev: float = Field(ge=0.0)


class PositionalRun(BaseModel):
    """Immutable raw artifact for a positional-retrieval execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    run_id: str
    recorded_at_utc: datetime
    dataset_sha256: str
    package_version: str
    python_version: str
    platform: str
    provider: str
    model: str
    profile: str
    max_context_tokens: int = Field(gt=0)
    requested_context_lengths: list[int]
    executed_context_lengths: list[int]
    skipped_context_lengths: list[int]
    strategies: list[PositionalStrategy]
    predictions: list[PositionalPrediction]
    accuracy_cells: list[PositionAccuracy]
    robustness: list[PositionalRobustness]
    metadata: dict[str, str | int | float | bool]
