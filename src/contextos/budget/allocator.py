"""Deterministic two-pass raw allocation and compression planning."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from contextos.budget.models import (
    AllocationPlan,
    CompressionRequest,
    ContextualBudget,
    DirectSelection,
)
from contextos.config import OptimizationPolicy
from contextos.errors import (
    AllocationError,
    ContextualBudgetInfeasible,
    InvalidScore,
    MandatoryContextOverflow,
)
from contextos.models import ContextItem, ContextType, validate_unique_item_ids
from contextos.scoring import ScoreBreakdown


@dataclass(frozen=True)
class _Candidate:
    item: ContextItem
    token_count: int
    utility: float
    value_density: float


def _token_count(item: ContextItem) -> int:
    if item.token_count is None:
        raise AllocationError(f"item {item.id} has not been tokenized")
    return item.token_count


def validate_contextual_budget(
    items: Sequence[ContextItem],
    *,
    policy: OptimizationPolicy,
) -> ContextualBudget:
    """Reserve mandatory tokens and validate only applicable class minima."""
    policy.validate_static()
    validate_unique_item_ids(items)
    mandatory_tokens = sum(_token_count(item) for item in items if item.mandatory)
    effective_budget = policy.effective_budget
    if mandatory_tokens > effective_budget:
        raise MandatoryContextOverflow(
            mandatory_tokens=mandatory_tokens,
            effective_budget=effective_budget,
        )
    optional_budget = effective_budget - mandatory_tokens
    present_types = tuple(
        context_type
        for context_type in ContextType
        if any(not item.mandatory and item.type is context_type for item in items)
    )
    applicable_minima = sum(
        policy.class_minimum_tokens.get(context_type, 0) for context_type in present_types
    )
    if applicable_minima > optional_budget:
        raise ContextualBudgetInfeasible(
            applicable_minima=applicable_minima,
            optional_budget=optional_budget,
        )
    return ContextualBudget(
        effective_budget=effective_budget,
        mandatory_tokens=mandatory_tokens,
        optional_budget=optional_budget,
        present_optional_types=present_types,
        applicable_minima=applicable_minima,
    )


def _candidate_rank(candidate: _Candidate) -> tuple[float, float, str]:
    return (-candidate.value_density, -candidate.utility, candidate.item.id)


def _compression_rank(
    candidate: _Candidate,
    target_tokens: int,
) -> tuple[float, float, str]:
    return (-(candidate.utility / target_tokens), -candidate.utility, candidate.item.id)


def _fits_class_maximum(
    candidate: _Candidate,
    allocated_by_type: Mapping[ContextType, int],
    policy: OptimizationPolicy,
    tokens: int,
) -> bool:
    maximum = policy.class_maximum_tokens.get(candidate.item.type)
    return maximum is None or allocated_by_type.get(candidate.item.type, 0) + tokens <= maximum


def _minimum_unmet_reason(
    candidates: Sequence[_Candidate],
    *,
    floor: int,
    remaining_budget: int,
    allocated_by_type: Mapping[ContextType, int],
    policy: OptimizationPolicy,
) -> str:
    if sum(candidate.token_count for candidate in candidates) < floor:
        return "no_more_candidates"
    if not any(candidate.token_count <= remaining_budget for candidate in candidates):
        return "no_candidate_fits_optional_budget"
    if not any(
        _fits_class_maximum(candidate, allocated_by_type, policy, candidate.token_count)
        for candidate in candidates
    ):
        return "class_maximum_prevents_floor"
    return "item_granularity_prevents_floor"


class TokenBudgetAllocator:
    """Build allocation plans without invoking a compressor."""

    def allocate(
        self,
        items: Sequence[ContextItem],
        *,
        scores: Mapping[str, ScoreBreakdown],
        policy: OptimizationPolicy,
    ) -> AllocationPlan:
        """Allocate tokenized optional items using the authoritative two-pass policy."""
        contextual = validate_contextual_budget(items, policy=policy)
        optional_items = [item for item in items if not item.mandatory]
        candidates: list[_Candidate] = []
        for item in optional_items:
            try:
                utility = scores[item.id].composite_utility
            except KeyError as exc:
                raise InvalidScore(f"composite score missing for optional item {item.id}") from exc
            token_count = _token_count(item)
            candidates.append(
                _Candidate(
                    item=item,
                    token_count=token_count,
                    utility=utility,
                    value_density=utility / max(token_count, 1),
                )
            )

        selected: list[DirectSelection] = []
        selected_ids: set[str] = set()
        allocated_by_type: dict[ContextType, int] = defaultdict(int)
        remaining_budget = contextual.optional_budget
        minimum_unmet: dict[ContextType, str] = {}

        for context_type in ContextType:
            floor = policy.class_minimum_tokens.get(context_type, 0)
            if floor == 0 or context_type not in contextual.present_optional_types:
                continue
            typed_candidates = sorted(
                (candidate for candidate in candidates if candidate.item.type is context_type),
                key=_candidate_rank,
            )
            for candidate in typed_candidates:
                if allocated_by_type[context_type] >= floor:
                    break
                if candidate.token_count > remaining_budget or not _fits_class_maximum(
                    candidate,
                    allocated_by_type,
                    policy,
                    candidate.token_count,
                ):
                    continue
                selected.append(
                    DirectSelection(
                        item_id=candidate.item.id,
                        allocated_tokens=candidate.token_count,
                        utility=candidate.utility,
                        value_density=candidate.value_density,
                        reason="class_soft_floor",
                    )
                )
                selected_ids.add(candidate.item.id)
                allocated_by_type[context_type] += candidate.token_count
                remaining_budget -= candidate.token_count
            if allocated_by_type[context_type] < floor:
                minimum_unmet[context_type] = _minimum_unmet_reason(
                    typed_candidates,
                    floor=floor,
                    remaining_budget=remaining_budget,
                    allocated_by_type=allocated_by_type,
                    policy=policy,
                )

        compression_pool: list[_Candidate] = []
        rejected_reasons: dict[str, str] = {}
        remaining_candidates = sorted(
            (candidate for candidate in candidates if candidate.item.id not in selected_ids),
            key=_candidate_rank,
        )
        for candidate in remaining_candidates:
            fits_budget = candidate.token_count <= remaining_budget
            fits_maximum = _fits_class_maximum(
                candidate,
                allocated_by_type,
                policy,
                candidate.token_count,
            )
            if fits_budget and fits_maximum:
                selected.append(
                    DirectSelection(
                        item_id=candidate.item.id,
                        allocated_tokens=candidate.token_count,
                        utility=candidate.utility,
                        value_density=candidate.value_density,
                        reason="global_value_density",
                    )
                )
                selected_ids.add(candidate.item.id)
                allocated_by_type[candidate.item.type] += candidate.token_count
                remaining_budget -= candidate.token_count
            elif policy.compression_enabled and candidate.item.compressible:
                compression_pool.append(candidate)
            elif not fits_maximum:
                rejected_reasons[candidate.item.id] = "class_maximum_exceeded"
            elif not candidate.item.compressible:
                rejected_reasons[candidate.item.id] = "not_compressible"
            else:
                rejected_reasons[candidate.item.id] = "compression_disabled"

        planned_candidates: list[tuple[_Candidate, int]] = []
        for candidate in compression_pool:
            target = max(
                policy.minimum_compressed_tokens,
                math.ceil(candidate.token_count * policy.compression_target_ratio),
            )
            if target >= candidate.token_count:
                rejected_reasons[candidate.item.id] = "compression_not_beneficial"
            else:
                planned_candidates.append((candidate, target))

        compression_budget = contextual.optional_budget - sum(
            selection.allocated_tokens for selection in selected
        )
        remaining_compression_budget = compression_budget
        compression_requests: list[CompressionRequest] = []
        ranked_compression_candidates = sorted(
            planned_candidates,
            key=lambda value: _compression_rank(value[0], value[1]),
        )
        for candidate, target in ranked_compression_candidates:
            if not _fits_class_maximum(candidate, allocated_by_type, policy, target):
                rejected_reasons[candidate.item.id] = "class_maximum_exceeded"
                continue
            if target > remaining_compression_budget:
                rejected_reasons[candidate.item.id] = "insufficient_compression_budget"
                continue
            compression_requests.append(
                CompressionRequest(
                    item_id=candidate.item.id,
                    target_tokens=target,
                    original_tokens=candidate.token_count,
                    utility=candidate.utility,
                    value_density=candidate.utility / target,
                    reason="raw_item_did_not_fit",
                )
            )
            allocated_by_type[candidate.item.type] += target
            remaining_compression_budget -= target

        candidate_ids = [candidate.item.id for candidate in candidates]
        rejected_ids = sorted(rejected_reasons)
        return AllocationPlan(
            direct_selected=selected,
            compression_requests=compression_requests,
            rejected_item_ids=rejected_ids,
            optional_budget=contextual.optional_budget,
            direct_tokens=sum(selection.allocated_tokens for selection in selected),
            compression_budget=compression_budget,
            candidate_item_ids=candidate_ids,
            rejection_reasons={item_id: rejected_reasons[item_id] for item_id in rejected_ids},
            minimum_unmet_reasons=minimum_unmet,
            class_allocated_tokens=dict(allocated_by_type),
            compression_candidate_order=[
                candidate.item.id for candidate, _ in ranked_compression_candidates
            ],
            compression_candidate_targets={
                candidate.item.id: target for candidate, target in ranked_compression_candidates
            },
        )
