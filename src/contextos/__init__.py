"""Public package surface for ContextOS."""

from contextos.baselines import (
    FullContextBaseline,
    LastNTokensBaseline,
    NaiveExtractiveBaseline,
    RelevanceOnlyBaseline,
    SlidingWindowBaseline,
)
from contextos.budget import AllocationPlan
from contextos.config import OptimizationPolicy
from contextos.models import (
    ContextEdge,
    ContextItem,
    ContextType,
    DependencyRelation,
    LifecycleTier,
)
from contextos.optimizer import ContextOptimizer
from contextos.trace import OptimizationTrace, OptimizedContext

__all__ = [
    "AllocationPlan",
    "ContextEdge",
    "ContextItem",
    "ContextOptimizer",
    "ContextType",
    "DependencyRelation",
    "FullContextBaseline",
    "LastNTokensBaseline",
    "LifecycleTier",
    "NaiveExtractiveBaseline",
    "OptimizationPolicy",
    "OptimizationTrace",
    "OptimizedContext",
    "RelevanceOnlyBaseline",
    "SlidingWindowBaseline",
]

__version__ = "0.3.0"
