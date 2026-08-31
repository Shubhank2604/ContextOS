"""Deterministic, non-destructive lifecycle transitions."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contextos.errors import LifecycleError
from contextos.models import ContextItem, ContextType, LifecycleTier
from contextos.store.base import ContextStore

_ALWAYS_HOT = {ContextType.SYSTEM_INSTRUCTION, ContextType.TASK_STATE}


class LifecyclePolicy(BaseModel):
    """Age and importance thresholds for storage-tier transitions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hot_seconds: float = Field(default=86_400.0, ge=0.0)
    warm_seconds: float = Field(default=604_800.0, ge=0.0)
    cold_seconds: float = Field(default=2_592_000.0, ge=0.0)
    hot_importance_threshold: float = Field(default=0.9, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> LifecyclePolicy:
        if not self.hot_seconds <= self.warm_seconds <= self.cold_seconds:
            raise ValueError("lifecycle age thresholds must be non-decreasing")
        return self


class LifecycleManager:
    """Compute and persist deterministic tiers without deleting records."""

    def __init__(self, policy: LifecyclePolicy | None = None) -> None:
        self.policy = policy or LifecyclePolicy()

    def tier_for(self, item: ContextItem, *, now: datetime) -> LifecycleTier:
        """Return the item's tier at an explicit point in time."""
        if now.tzinfo is None or now.utcoffset() is None:
            raise LifecycleError("lifecycle evaluation time must be timezone-aware")
        if item.lifecycle_tier is LifecycleTier.ARCHIVED:
            return LifecycleTier.ARCHIVED
        override = item.metadata.get("lifecycle_tier_override")
        if override is not None:
            try:
                return LifecycleTier(str(override))
            except ValueError as exc:
                raise LifecycleError(f"invalid lifecycle tier override: {override!r}") from exc
        if item.type in _ALWAYS_HOT or item.importance >= self.policy.hot_importance_threshold:
            return LifecycleTier.HOT
        reference = self._last_selected(item)
        age = max((now - reference).total_seconds(), 0.0)
        if age <= self.policy.hot_seconds:
            return LifecycleTier.HOT
        if age <= self.policy.warm_seconds:
            return LifecycleTier.WARM
        if age <= self.policy.cold_seconds:
            return LifecycleTier.COLD
        return LifecycleTier.ARCHIVED

    def transition_store(self, store: ContextStore, *, now: datetime) -> dict[str, LifecycleTier]:
        """Update every changed tier and return the applied transitions."""
        transitions: dict[str, LifecycleTier] = {}
        for item in store.list_items():
            target = self.tier_for(item, now=now)
            if target is not item.lifecycle_tier:
                store.update_tier(item.id, target)
                transitions[item.id] = target
        return transitions

    @staticmethod
    def _last_selected(item: ContextItem) -> datetime:
        value = item.metadata.get("last_selected_at")
        if value is None:
            return item.updated_at
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError as exc:
                raise LifecycleError(f"invalid last_selected_at for {item.id!r}") from exc
        else:
            raise LifecycleError(f"invalid last_selected_at for {item.id!r}")
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise LifecycleError(f"last_selected_at for {item.id!r} must be timezone-aware")
        return parsed
