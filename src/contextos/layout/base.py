"""Layout strategy contract."""

from collections.abc import Mapping, Sequence
from typing import Protocol

from contextos.models import ContextItem
from contextos.scoring import ScoreBreakdown


class LayoutStrategy(Protocol):
    """Arrange selected items without changing their content."""

    def arrange(
        self,
        items: Sequence[ContextItem],
        *,
        scores: Mapping[str, ScoreBreakdown],
        original_positions: Mapping[str, int],
    ) -> list[ContextItem]:
        """Return a deterministic permutation of the selected items."""
        ...
