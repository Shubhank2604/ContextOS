"""Offline exact key-value provider for benchmark plumbing validation."""

from __future__ import annotations

import re

from contextos.providers.base import ProviderResponse

_QUERY_PATTERN = re.compile(r"^QUERY_KEY: ([A-Z0-9_-]+)$", re.MULTILINE)
_RECORD_PATTERN = re.compile(
    r"^RECORD ([A-Z0-9_-]+) => ([A-Z0-9_-]+)$",
    re.MULTILINE,
)


class DeterministicRetrievalProvider:
    """Resolve the queried record exactly, independent of its prompt position."""

    model = "deterministic-retrieval-v1"

    def complete(self, prompt: str, *, max_output_tokens: int) -> ProviderResponse:
        """Return the matching value or an empty prediction when none exists."""
        del max_output_tokens
        query = _QUERY_PATTERN.search(prompt)
        if query is None:
            return ProviderResponse(text="", model=self.model)
        values = dict(_RECORD_PATTERN.findall(prompt))
        return ProviderResponse(text=values.get(query.group(1), ""), model=self.model)
