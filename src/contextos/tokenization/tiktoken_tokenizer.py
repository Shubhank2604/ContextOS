"""Default token counting backed by tiktoken."""

from __future__ import annotations

import tiktoken
from tiktoken import Encoding

from contextos.config import DEFAULT_TIKTOKEN_ENCODING
from contextos.errors import TokenizerError


class TiktokenTokenizer:
    """Deterministic token counter for a configured tiktoken encoding."""

    def __init__(self, encoding_name: str = DEFAULT_TIKTOKEN_ENCODING) -> None:
        self.encoding_name = encoding_name
        try:
            self._encoding: Encoding = tiktoken.get_encoding(encoding_name)
        except Exception as exc:
            raise TokenizerError(f"Unable to load tiktoken encoding {encoding_name!r}") from exc

    def count_tokens(self, text: str) -> int:
        """Return the encoded token count, wrapping tokenizer failures."""
        try:
            return len(self._encoding.encode(text))
        except Exception as exc:
            raise TokenizerError("Unable to tokenize context content") from exc
