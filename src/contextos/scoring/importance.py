"""Application-supplied importance scoring."""

from __future__ import annotations

from collections.abc import Sequence

from contextos.models import ContextItem


def importance_scores(items: Sequence[ContextItem]) -> dict[str, float]:
    """Return validated application importance or model-applied type defaults."""
    return {item.id: item.importance for item in items}
