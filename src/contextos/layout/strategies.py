"""Deterministic context layout strategies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from contextos.errors import InvalidScore
from contextos.models import ContextItem, ContextType, validate_unique_item_ids
from contextos.scoring import ScoreBreakdown

_TRAILING_TYPES = {ContextType.TASK_STATE, ContextType.USER_MESSAGE}


def _validate_layout_inputs(
    items: Sequence[ContextItem],
    original_positions: Mapping[str, int],
) -> None:
    validate_unique_item_ids(items)
    missing = sorted(item.id for item in items if item.id not in original_positions)
    if missing:
        raise ValueError(f"original positions missing for: {', '.join(missing)}")


class OriginalOrderLayout:
    """Restore the source order supplied to the optimizer."""

    def arrange(
        self,
        items: Sequence[ContextItem],
        *,
        scores: Mapping[str, ScoreBreakdown],
        original_positions: Mapping[str, int],
    ) -> list[ContextItem]:
        """Order solely by original position with ID as a stable tie-breaker."""
        del scores
        _validate_layout_inputs(items, original_positions)
        return sorted(items, key=lambda item: (original_positions[item.id], item.id))


class RelevanceDescendingLayout:
    """Place the most task-relevant items first."""

    def arrange(
        self,
        items: Sequence[ContextItem],
        *,
        scores: Mapping[str, ScoreBreakdown],
        original_positions: Mapping[str, int],
    ) -> list[ContextItem]:
        """Sort by relevance and then source order deterministically."""
        _validate_layout_inputs(items, original_positions)
        missing = sorted(item.id for item in items if item.id not in scores)
        if missing:
            raise InvalidScore(f"layout score missing for: {', '.join(missing)}")
        return sorted(
            items,
            key=lambda item: (
                -scores[item.id].relevance,
                original_positions[item.id],
                item.id,
            ),
        )


class PositionAwareLayout:
    """Use explicit beginning, middle, and recency-sensitive end regions."""

    def arrange(
        self,
        items: Sequence[ContextItem],
        *,
        scores: Mapping[str, ScoreBreakdown],
        original_positions: Mapping[str, int],
    ) -> list[ContextItem]:
        """Place system rules first, high-value evidence early, and current state late."""
        _validate_layout_inputs(items, original_positions)
        optional = [item for item in items if not item.mandatory]
        missing = sorted(item.id for item in optional if item.id not in scores)
        if missing:
            raise InvalidScore(f"layout score missing for: {', '.join(missing)}")

        mandatory_system = sorted(
            (
                item
                for item in items
                if item.mandatory and item.type is ContextType.SYSTEM_INSTRUCTION
            ),
            key=lambda item: (original_positions[item.id], item.id),
        )
        mandatory_other = sorted(
            (
                item
                for item in items
                if item.mandatory and item.type is not ContextType.SYSTEM_INSTRUCTION
            ),
            key=lambda item: (original_positions[item.id], item.id),
        )
        trailing = sorted(
            (item for item in optional if item.type in _TRAILING_TYPES),
            key=lambda item: (item.updated_at, original_positions[item.id], item.id),
        )
        evidence = sorted(
            (item for item in optional if item.type not in _TRAILING_TYPES),
            key=lambda item: (
                -scores[item.id].composite_utility,
                original_positions[item.id],
                item.id,
            ),
        )
        primary_count = (len(evidence) + 1) // 2
        primary = evidence[:primary_count]
        secondary = evidence[primary_count:]
        return [*mandatory_system, *mandatory_other, *primary, *secondary, *trailing]
