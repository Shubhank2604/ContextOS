"""Validated schemas for the configured LongBench external-validation subset."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LongBenchProfile(StrEnum):
    """Deterministic external-benchmark execution profiles."""

    QUICK = "quick"
    STANDARD = "standard"
    FULL = "full"


class LongBenchMetric(StrEnum):
    """Dataset-appropriate deterministic metrics used by the selected tasks."""

    QA_F1 = "qa_f1"
    RETRIEVAL = "retrieval"
    CODE_SIMILARITY = "code_similarity"


class LongBenchTaskConfig(BaseModel):
    """Pinned configuration for one upstream LongBench task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    family: str
    metric: LongBenchMetric
    prompt_template: str
    max_output_tokens: int = Field(gt=0)
    quick_samples: int = Field(gt=0)
    standard_samples: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_prompt_template(self) -> LongBenchTaskConfig:
        if "{context}" not in self.prompt_template or "{input}" not in self.prompt_template:
            raise ValueError("LongBench prompt templates must contain {context} and {input}")
        return self


class LongBenchSubsetConfig(BaseModel):
    """Versioned source, sampling, and metric configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    source_repository: str
    source_revision: str
    source_split: str
    sampling_seed: int = Field(ge=0)
    tasks: list[LongBenchTaskConfig]

    @model_validator(mode="after")
    def validate_tasks(self) -> LongBenchSubsetConfig:
        names = [task.dataset for task in self.tasks]
        if not self.tasks or len(names) != len(set(names)):
            raise ValueError("LongBench task names must be non-empty and unique")
        if sum(task.standard_samples for task in self.tasks) < 100:
            raise ValueError("LongBench standard profile must configure at least 100 examples")
        return self


class LongBenchCase(BaseModel):
    """One preserved upstream example with its original source identifier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    source_id: str
    input: str
    context: str
    answers: list[str]
    source_length: int = Field(ge=0)
    language: str
    all_classes: list[str] | None = None
    metric: LongBenchMetric
    prompt_template: str
    max_output_tokens: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_content(self) -> LongBenchCase:
        if not self.source_id.strip() or not self.input.strip() or not self.context.strip():
            raise ValueError("LongBench source ID, input, and context must not be blank")
        if not self.answers or any(not answer.strip() for answer in self.answers):
            raise ValueError("LongBench answers must be non-empty strings")
        return self


class PreparedLongBenchSubset(BaseModel):
    """Deterministically selected, locally materialized external cases."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    profile: LongBenchProfile
    source_repository: str
    source_revision: str
    source_split: str
    sampling_seed: int = Field(ge=0)
    cases: list[LongBenchCase]

    @model_validator(mode="after")
    def validate_case_identity(self) -> PreparedLongBenchSubset:
        identities = [(case.dataset, case.source_id) for case in self.cases]
        if not self.cases or len(identities) != len(set(identities)):
            raise ValueError("prepared LongBench case identities must be non-empty and unique")
        return self


class LongBenchPrediction(BaseModel):
    """Provider output keyed by the preserved upstream example identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    source_id: str
    strategy: str = "unassigned"
    status: str = "ok"
    prediction: str
    provider: str
    model: str
    original_context_tokens: int | None = Field(default=None, ge=0)
    input_context_tokens: int | None = Field(default=None, ge=0)
    prompt_input_tokens: int | None = Field(default=None, ge=0)
    provider_input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    context_reduction: float | None = Field(default=None, ge=0.0, le=1.0)
    optimizer_latency_ms: float | None = Field(default=None, ge=0.0)
    provider_latency_ms: float | None = Field(default=None, ge=0.0)
    selected_item_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identity(self) -> LongBenchPrediction:
        if not all(
            value.strip()
            for value in (
                self.dataset,
                self.source_id,
                self.strategy,
                self.status,
                self.provider,
                self.model,
            )
        ):
            raise ValueError("prediction dataset, source ID, provider, and model must not be blank")
        return self


class LongBenchCaseScore(BaseModel):
    """Raw deterministic score for one external example."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    source_id: str
    strategy: str
    status: str
    metric: LongBenchMetric
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    quality_retention: float | None = Field(default=None, ge=0.0)
    prediction: str
    answers: list[str]


class LongBenchDatasetAggregate(BaseModel):
    """Aggregate retained separately because task metrics are not interchangeable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    strategy: str
    metric: LongBenchMetric
    case_count: int = Field(gt=0)
    successful_case_count: int = Field(ge=0)
    mean_score: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_quality_retention: float | None = Field(default=None, ge=0.0)


class LongBenchScoreReport(BaseModel):
    """Machine-readable external benchmark evaluation output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    prepared_sha256: str
    prediction_count: int = Field(gt=0)
    provider: str
    model: str
    case_scores: list[LongBenchCaseScore]
    dataset_aggregates: list[LongBenchDatasetAggregate]
