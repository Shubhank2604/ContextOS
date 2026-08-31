"""Public package surface for ContextOS."""

from contextos.baselines import FullContextBaseline, LastNTokensBaseline, SlidingWindowBaseline
from contextos.config import OptimizationPolicy
from contextos.models import ContextItem, ContextType, LifecycleTier
from contextos.trace import OptimizationTrace, OptimizedContext

__all__ = [
    "ContextItem",
    "ContextType",
    "FullContextBaseline",
    "LastNTokensBaseline",
    "LifecycleTier",
    "OptimizationPolicy",
    "OptimizationTrace",
    "OptimizedContext",
    "SlidingWindowBaseline",
]

__version__ = "0.2.0"
