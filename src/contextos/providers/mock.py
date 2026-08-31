"""Deterministic provider for tests and offline evaluation."""

from contextos.providers.base import ProviderResponse


class MockLLMProvider:
    """Return a configured response without network access."""

    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, prompt: str, *, max_output_tokens: int) -> ProviderResponse:
        """Return the configured response and retain no prompt state."""
        del prompt, max_output_tokens
        return ProviderResponse(text=self._response, model="mock")
