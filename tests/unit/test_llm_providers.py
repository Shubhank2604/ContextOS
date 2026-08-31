"""Offline-safe optional language-model provider tests."""

import pytest

from contextos.errors import LLMProviderError
from contextos.providers import MockLLMProvider, OpenAIProvider


def test_mock_provider_is_deterministic() -> None:
    result = MockLLMProvider("fixed").complete("ignored", max_output_tokens=4)
    assert result.text == "fixed"
    assert result.model == "mock"


def test_openai_provider_requires_explicit_model_and_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        OpenAIProvider(model=" ")
    with pytest.raises(LLMProviderError, match="OPENAI_API_KEY"):
        OpenAIProvider(model="configured-benchmark-model").complete("prompt", max_output_tokens=8)
