"""Deterministic exponential-recency scoring."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from contextos.models import ContextItem


def recency_scores(
    items: Sequence[ContextItem],
    *,
    half_life_seconds: float,
    reference_time: datetime | None = None,
) -> dict[str, float]:
    """Apply ``0.5 ** (age / half_life)`` using a deterministic reference."""
    if half_life_seconds <= 0:
        raise ValueError("half_life_seconds must be positive")
    if not items:
        return {}
    effective_reference = reference_time or max(item.updated_at for item in items)
    if effective_reference.tzinfo is None or effective_reference.utcoffset() is None:
        raise ValueError("reference_time must be timezone-aware")
    return {
        item.id: 0.5
        ** (max((effective_reference - item.updated_at).total_seconds(), 0.0) / half_life_seconds)
        for item in items
    }
