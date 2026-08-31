"""Deterministic, bounded, cycle-safe dependency propagation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from contextos.errors import InvalidScore, UnknownDependencyReference
from contextos.models import ContextEdge


@dataclass(frozen=True)
class _Neighbor:
    item_id: str
    edge: ContextEdge


class DependencyGraph:
    """Validated context dependency graph with deterministic traversal."""

    def __init__(self, item_ids: Sequence[str], edges: Sequence[ContextEdge]) -> None:
        self.item_ids = frozenset(item_ids)
        adjacency: dict[str, list[_Neighbor]] = defaultdict(list)
        for edge in edges:
            unknown = sorted({edge.source_id, edge.target_id} - self.item_ids)
            if unknown:
                raise UnknownDependencyReference(
                    "dependency references unknown item IDs: " + ", ".join(unknown)
                )
            adjacency[edge.source_id].append(_Neighbor(edge.target_id, edge))
            if edge.source_id != edge.target_id:
                adjacency[edge.target_id].append(_Neighbor(edge.source_id, edge))
        self._adjacency = {
            item_id: sorted(
                neighbors,
                key=lambda neighbor: (
                    neighbor.item_id,
                    neighbor.edge.relation.value,
                    neighbor.edge.source_id,
                    neighbor.edge.target_id,
                ),
            )
            for item_id, neighbors in adjacency.items()
        }
        self.edges = tuple(
            sorted(
                edges,
                key=lambda edge: (
                    edge.source_id,
                    edge.target_id,
                    edge.relation.value,
                    edge.weight,
                ),
            )
        )

    def edges_for(self, item_id: str) -> tuple[ContextEdge, ...]:
        """Return every relation touching an item in deterministic order."""
        if item_id not in self.item_ids:
            raise UnknownDependencyReference(f"unknown dependency item ID: {item_id}")
        return tuple(neighbor.edge for neighbor in self._adjacency.get(item_id, ()))

    def propagate_scores(
        self,
        base_scores: Mapping[str, float],
        *,
        max_depth: int = 2,
    ) -> dict[str, float]:
        """Propagate the strongest weighted neighbor score up to ``max_depth``."""
        if type(max_depth) is not int or max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer")
        missing = sorted(self.item_ids - set(base_scores))
        unknown = sorted(set(base_scores) - self.item_ids)
        if missing or unknown:
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if unknown:
                details.append("unknown: " + ", ".join(unknown))
            raise InvalidScore(
                "dependency base-score IDs do not match graph IDs (" + "; ".join(details) + ")"
            )
        for item_id, score in base_scores.items():
            if not 0.0 <= score <= 1.0:
                raise InvalidScore(f"base score for {item_id} must be between 0 and 1")

        propagated: dict[str, float] = {}
        for origin in sorted(self.item_ids):
            best = 0.0
            stack: list[tuple[str, int, float, frozenset[str]]] = [
                (origin, 0, 1.0, frozenset({origin}))
            ]
            while stack:
                current, depth, path_weight, visited = stack.pop()
                if depth >= max_depth:
                    continue
                for neighbor in reversed(self._adjacency.get(current, ())):
                    if neighbor.item_id in visited:
                        continue
                    next_weight = path_weight * neighbor.edge.weight
                    best = max(best, base_scores[neighbor.item_id] * next_weight)
                    stack.append(
                        (
                            neighbor.item_id,
                            depth + 1,
                            next_weight,
                            visited | {neighbor.item_id},
                        )
                    )
            propagated[origin] = min(max(best, 0.0), 1.0)
        return propagated
