"""Public package surface for ContextOS."""

from contextos.baselines import FullContextBaseline, LastNTokensBaseline, SlidingWindowBaseline
from contextos.budget import AllocationPlan
from contextos.config import OptimizationPolicy
from contextos.models import (
    ContextEdge,
    ContextItem,
    ContextType,
    DependencyRelation,
    LifecycleTier,
)
from contextos.trace import OptimizationTrace, OptimizedContext

__all__ = [
    "AllocationPlan",
    "ContextEdge",
    "ContextItem",
    "ContextType",
    "DependencyRelation",
    "FullContextBaseline",
    "LastNTokensBaseline",
    "LifecycleTier",
    "OptimizationPolicy",
    "OptimizationTrace",
    "OptimizedContext",
    "SlidingWindowBaseline",
]

__version__ = "0.2.0"
