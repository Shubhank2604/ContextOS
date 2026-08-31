"""Explicit allocator-to-compressor handoff models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contextos.models import ContextType


class DirectSelection(BaseModel):
    """An optional item selected without compression."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str
    allocated_tokens: int = Field(ge=0)
    utility: float = Field(ge=0.0, le=1.0)
    value_density: float = Field(ge=0.0)
    reason: str


class CompressionRequest(BaseModel):
    """A reserved target budget for later independent compression."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str
    target_tokens: int = Field(gt=0)
    original_tokens: int = Field(gt=0)
    utility: float = Field(ge=0.0, le=1.0)
    value_density: float = Field(ge=0.0)
    reason: str

    @model_validator(mode="after")
    def validate_budget_benefit(self) -> CompressionRequest:
        """Reject requests that cannot reduce the original token count."""
        if self.target_tokens >= self.original_tokens:
            raise ValueError("compression target must be smaller than original tokens")
        return self


class ContextualBudget(BaseModel):
    """Budget remaining after mandatory reservation and minima validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    effective_budget: int = Field(gt=0)
    mandatory_tokens: int = Field(ge=0)
    optional_budget: int = Field(ge=0)
    present_optional_types: tuple[ContextType, ...]
    applicable_minima: int = Field(ge=0)


class AllocationPlan(BaseModel):
    """Complete deterministic outcome for every optional allocation candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    direct_selected: list[DirectSelection]
    compression_requests: list[CompressionRequest]
    rejected_item_ids: list[str]
    optional_budget: int = Field(ge=0)
    direct_tokens: int = Field(ge=0)
    compression_budget: int = Field(ge=0)

    candidate_item_ids: list[str]
    rejection_reasons: dict[str, str]
    minimum_unmet_reasons: dict[ContextType, str] = Field(default_factory=dict)
    class_allocated_tokens: dict[ContextType, int] = Field(default_factory=dict)
    compression_candidate_order: list[str] = Field(default_factory=list)
    compression_candidate_targets: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_partition_and_accounting(self) -> AllocationPlan:
        """Enforce one outcome per candidate and budget accounting."""
        direct_ids = [selection.item_id for selection in self.direct_selected]
        compression_ids = [request.item_id for request in self.compression_requests]
        outcome_ids = [*direct_ids, *compression_ids, *self.rejected_item_ids]
        if len(outcome_ids) != len(set(outcome_ids)):
            raise ValueError("an allocation candidate appears in more than one outcome")
        if set(outcome_ids) != set(self.candidate_item_ids):
            raise ValueError("every allocation candidate must appear in exactly one outcome")
        if len(self.candidate_item_ids) != len(set(self.candidate_item_ids)):
            raise ValueError("candidate item IDs must be unique")
        if set(self.rejection_reasons) != set(self.rejected_item_ids):
            raise ValueError("every rejected item must have exactly one rejection reason")
        if self.direct_tokens != sum(
            selection.allocated_tokens for selection in self.direct_selected
        ):
            raise ValueError("direct_tokens does not match direct selections")
        if self.compression_budget != self.optional_budget - self.direct_tokens:
            raise ValueError("compression_budget must equal optional_budget - direct_tokens")
        reserved = sum(request.target_tokens for request in self.compression_requests)
        if reserved > self.compression_budget:
            raise ValueError("compression reservations exceed compression budget")
        if len(self.compression_candidate_order) != len(set(self.compression_candidate_order)):
            raise ValueError("compression candidate order must contain unique IDs")
        if set(self.compression_candidate_targets) != set(self.compression_candidate_order):
            raise ValueError("every ranked compression candidate must have one target")
        if any(target <= 0 for target in self.compression_candidate_targets.values()):
            raise ValueError("compression candidate targets must be positive")
        return self

    @property
    def reserved_compression_tokens(self) -> int:
        """Return planned compression tokens currently reserved."""
        return sum(request.target_tokens for request in self.compression_requests)

    @property
    def remaining_compression_tokens(self) -> int:
        """Return unreserved compression budget."""
        return self.compression_budget - self.reserved_compression_tokens
