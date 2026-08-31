"""Tests for validated, deterministic dependency propagation."""

import pytest

from contextos.dependency import DependencyGraph
from contextos.errors import InvalidScore, UnknownDependencyReference
from contextos.models import ContextEdge, DependencyRelation


def _edge(
    source: str,
    target: str,
    *,
    relation: DependencyRelation = DependencyRelation.REQUIRES,
    weight: float = 1.0,
) -> ContextEdge:
    return ContextEdge(source_id=source, target_id=target, relation=relation, weight=weight)


def test_unknown_dependency_endpoint_fails_validation() -> None:
    with pytest.raises(UnknownDependencyReference, match="missing"):
        DependencyGraph(["known"], [_edge("known", "missing")])


def test_dependency_propagation_respects_max_depth_and_path_weights() -> None:
    graph = DependencyGraph(
        ["a", "b", "c"],
        [_edge("a", "b", weight=0.5), _edge("b", "c", weight=0.5)],
    )
    base = {"a": 0.0, "b": 0.0, "c": 1.0}
    assert graph.propagate_scores(base, max_depth=1)["a"] == 0.0
    assert graph.propagate_scores(base, max_depth=2)["a"] == pytest.approx(0.25)


def test_dependency_cycles_terminate_deterministically() -> None:
    graph = DependencyGraph(
        ["a", "b", "c"],
        [_edge("a", "b"), _edge("b", "c"), _edge("c", "a")],
    )
    base = {"a": 0.1, "b": 0.5, "c": 0.9}
    first = graph.propagate_scores(base, max_depth=20)
    second = graph.propagate_scores(base, max_depth=20)
    assert first == second
    assert first == {"a": 0.9, "b": 0.9, "c": 0.5}


def test_supersedes_and_contradicts_edges_remain_traceable() -> None:
    supersedes = _edge("new", "old", relation=DependencyRelation.SUPERSEDES)
    contradicts = _edge("new", "other", relation=DependencyRelation.CONTRADICTS)
    graph = DependencyGraph(["new", "old", "other"], [contradicts, supersedes])
    assert graph.edges_for("new") == (supersedes, contradicts)
    assert {edge.relation for edge in graph.edges} == {
        DependencyRelation.SUPERSEDES,
        DependencyRelation.CONTRADICTS,
    }


@pytest.mark.parametrize("max_depth", [-1, True])
def test_dependency_depth_must_be_non_negative_integer(max_depth: int) -> None:
    graph = DependencyGraph(["a"], [])
    with pytest.raises(ValueError, match="non-negative integer"):
        graph.propagate_scores({"a": 0.5}, max_depth=max_depth)


@pytest.mark.parametrize(
    "scores",
    [
        {},
        {"a": 0.5, "unknown": 0.2},
        {"a": -0.1},
        {"a": 1.1},
    ],
)
def test_dependency_base_scores_must_match_graph_contract(scores: dict[str, float]) -> None:
    graph = DependencyGraph(["a"], [])
    with pytest.raises(InvalidScore):
        graph.propagate_scores(scores)
