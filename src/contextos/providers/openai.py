"""Lazy optional OpenAI Responses API provider for explicit benchmark runs."""

from __future__ import annotations

import importlib
import os
from typing import Any

from contextos.errors import LLMProviderError
from contextos.providers.base import ProviderResponse


class OpenAIProvider:
    """Call OpenAI only when ``complete`` is explicitly invoked."""

    def __init__(self, *, model: str, temperature: float | None = None) -> None:
        if not model.strip():
            raise ValueError("OpenAI provider model must not be empty")
        if temperature is not None and not 0.0 <= temperature <= 2.0:
            raise ValueError("OpenAI provider temperature must be between zero and two")
        self.model = model
        self.temperature = temperature

    def complete(self, prompt: str, *, max_output_tokens: int) -> ProviderResponse:
        """Generate one bounded response using an environment-provided API key."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMProviderError(
                "OPENAI_API_KEY is required only for an explicitly requested OpenAI run"
            )
        try:
            module = importlib.import_module("openai")
        except ImportError as exc:
            raise LLMProviderError(
                "Install the 'openai' optional dependency to run OpenAI benchmarks"
            ) from exc
        try:
            client_type: Any = module.OpenAI
            client = client_type(api_key=api_key)
            request: dict[str, Any] = {
                "model": self.model,
                "input": prompt,
                "max_output_tokens": max_output_tokens,
            }
            if self.temperature is not None:
                request["temperature"] = self.temperature
            response = client.responses.create(**request)
            usage = getattr(response, "usage", None)
            input_details = getattr(usage, "input_tokens_details", None)
            return ProviderResponse(
                text=str(response.output_text),
                input_tokens=getattr(usage, "input_tokens", None),
                output_tokens=getattr(usage, "output_tokens", None),
                cached_tokens=getattr(input_details, "cached_tokens", None),
                model=self.model,
            )
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(f"OpenAI response failed: {exc}") from exc
