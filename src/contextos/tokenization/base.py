"""Token-counting interface."""

from typing import Protocol


class Tokenizer(Protocol):
    """Counts tokens according to a configured encoding."""

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in text."""
        ...
