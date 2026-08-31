"""Compression contracts shared by all strategies."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contextos.models import ContextItem


class CompressionResult(BaseModel):
    """Auditable output from one compression attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str | None = None
    original_tokens: int = Field(ge=0)
    compressed_tokens: int = Field(ge=0)
    source_item_id: str
    strategy: str
    provenance: tuple[str, ...]
    failure_reason: str | None = None
    lossy: bool = False

    @model_validator(mode="after")
    def validate_result(self) -> CompressionResult:
        """Require successful results to be non-empty and source-attributed."""
        if self.source_item_id not in self.provenance:
            raise ValueError("compression provenance must include the source item ID")
        if self.failure_reason is None:
            if self.content is None or not self.content.strip():
                raise ValueError("successful compression must return non-empty content")
            if self.compressed_tokens > self.original_tokens:
                raise ValueError("compression cannot increase the token count")
        return self

    @property
    def succeeded(self) -> bool:
        """Return whether this attempt produced usable content."""
        return self.failure_reason is None


class Compressor(Protocol):
    """Independent item compressor."""

    def compress(
        self,
        item: ContextItem,
        target_tokens: int,
        task: str,
    ) -> CompressionResult:
        """Compress one item without mutating it."""
        ...
