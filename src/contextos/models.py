"""Validated core data models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contextos.errors import DuplicateContextItemError


class ContextType(str, Enum):
    """Supported categories of heterogeneous runtime context."""

    SYSTEM_INSTRUCTION = "system_instruction"
    TOOL_DEFINITION = "tool_definition"
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_OUTPUT = "tool_output"
    RETRIEVED_DOCUMENT = "retrieved_document"
    MEMORY = "memory"
    DECISION = "decision"
    ERROR = "error"
    PLAN = "plan"
    CODE = "code"
    TASK_STATE = "task_state"


class LifecycleTier(str, Enum):
    """Persistence and selection lifecycle for a context item."""

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    ARCHIVED = "archived"


class DependencyRelation(str, Enum):
    """Supported deterministic relationships between context items."""

    REQUIRES = "requires"
    DERIVED_FROM = "derived_from"
    SUPERSEDES = "supersedes"
    CONTRADICTS = "contradicts"
    RELATED_TO = "related_to"


DEFAULT_IMPORTANCE_BY_TYPE: dict[ContextType, float] = {
    ContextType.SYSTEM_INSTRUCTION: 1.0,
    ContextType.TOOL_DEFINITION: 0.7,
    ContextType.USER_MESSAGE: 0.7,
    ContextType.ASSISTANT_MESSAGE: 0.4,
    ContextType.TOOL_OUTPUT: 0.4,
    ContextType.RETRIEVED_DOCUMENT: 0.5,
    ContextType.MEMORY: 0.5,
    ContextType.DECISION: 0.8,
    ContextType.ERROR: 0.8,
    ContextType.PLAN: 0.6,
    ContextType.CODE: 0.7,
    ContextType.TASK_STATE: 0.9,
}


class ContextEdge(BaseModel):
    """An application-supplied dependency edge between context items."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    target_id: str
    relation: DependencyRelation
    weight: float = Field(ge=0.0, le=1.0)

    @field_validator("source_id", "target_id")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        """Reject blank edge endpoints."""
        if not value.strip():
            raise ValueError("dependency endpoint IDs must not be empty")
        return value


class ContextItem(BaseModel):
    """A typed, validated unit of candidate LLM context."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str
    content: str
    type: ContextType
    source: str | None = None

    created_at: datetime
    updated_at: datetime

    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    mandatory: bool = False
    compressible: bool = True
    evictable: bool = True

    lifecycle_tier: LifecycleTier = LifecycleTier.HOT

    token_count: int | None = Field(default=None, ge=0)
    semantic_hash: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def apply_default_importance(cls, data: object) -> object:
        """Apply a deterministic type fallback only when importance is omitted."""
        if isinstance(data, Mapping) and "importance" not in data:
            values = dict(data)
            try:
                context_type = ContextType(values.get("type"))
            except (TypeError, ValueError):
                return data
            values["importance"] = DEFAULT_IMPORTANCE_BY_TYPE[context_type]
            return values
        return data

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        """Reject empty or whitespace-only identifiers."""
        if not value.strip():
            raise ValueError("id must not be empty")
        return value

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """Reject empty content; metadata-only records are not part of v1."""
        if not value.strip():
            raise ValueError("content must not be empty")
        return value

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_timezone_aware(cls, value: datetime) -> datetime:
        """Require timestamps with a meaningful UTC offset."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_invariants(self) -> ContextItem:
        """Enforce relationships between model fields."""
        if self.mandatory and self.evictable:
            raise ValueError("mandatory context cannot be evictable")
        return self

    def __setattr__(self, name: str, value: object) -> None:
        """Preserve invariants and invalidate derived fields before mutation."""
        if name == "mandatory" and value is True and self.__dict__.get("evictable") is True:
            raise ValueError("mandatory context cannot be evictable")
        if name == "evictable" and value is True and self.__dict__.get("mandatory") is True:
            raise ValueError("mandatory context cannot be evictable")

        content_changed = name == "content" and "content" in self.__dict__
        if content_changed:
            previous_content = self.__dict__["content"]
            super().__setattr__(name, value)
            if value != previous_content:
                super().__setattr__("token_count", None)
                super().__setattr__("semantic_hash", None)
            return
        super().__setattr__(name, value)


def validate_unique_item_ids(items: Sequence[ContextItem]) -> None:
    """Raise when a context collection contains duplicate item IDs."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in items:
        if item.id in seen:
            duplicates.add(item.id)
        seen.add(item.id)
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise DuplicateContextItemError(f"Duplicate context item IDs: {duplicate_list}")
