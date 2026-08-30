"""Tests for deterministic token counting."""

import pytest

from contextos.errors import TokenizerError
from contextos.tokenization import TiktokenTokenizer


def test_tiktoken_count_is_deterministic() -> None:
    tokenizer = TiktokenTokenizer()
    text = "Context construction must preserve required facts."
    assert tokenizer.count_tokens(text) == tokenizer.count_tokens(text)
    assert tokenizer.count_tokens(text) > 0


def test_empty_text_has_zero_tokens() -> None:
    assert TiktokenTokenizer().count_tokens("") == 0


def test_unknown_encoding_raises_typed_error() -> None:
    with pytest.raises(TokenizerError, match="Unable to load"):
        TiktokenTokenizer("not-a-real-encoding")
