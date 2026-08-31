"""Configurable deterministic context-type priorities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from contextos.config import DEFAULT_TYPE_PRIORITIES
from contextos.models import ContextItem, ContextType


def type_priority_scores(
    items: Sequence[ContextItem],
    *,
    priorities: Mapping[ContextType, float] | None = None,
) -> dict[str, float]:
    """Return configured priority with documented defaults for missing types."""
    configured = priorities or {}
    return {
        item.id: configured.get(item.type, DEFAULT_TYPE_PRIORITIES[item.type]) for item in items
    }
