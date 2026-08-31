"""Optional language-model provider contracts."""

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class ProviderResponse(BaseModel):
    """Provider-neutral text generation response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    model: str | None = None


class LLMProvider(Protocol):
    """Minimal interface required by optional lossy summarization."""

    def complete(self, prompt: str, *, max_output_tokens: int) -> ProviderResponse:
        """Generate bounded text or raise a typed provider error."""
        ...
