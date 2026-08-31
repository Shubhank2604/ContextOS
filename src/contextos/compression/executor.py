"""Deterministic execution of allocator compression reservations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from contextos.budget import AllocationPlan
from contextos.compression.base import CompressionResult, Compressor
from contextos.compression.extractive import ExtractiveCompressor
from contextos.compression.none import NoneCompressor
from contextos.compression.tool_output import ToolOutputCompressor
from contextos.config import OptimizationPolicy
from contextos.models import ContextItem, ContextType
from contextos.tokenization import Tokenizer

_TOOL_TYPES = {ContextType.TOOL_OUTPUT, ContextType.ERROR}
_PROTECTED_TYPES = {
    ContextType.SYSTEM_INSTRUCTION,
    ContextType.TOOL_DEFINITION,
    ContextType.CODE,
}


class CompressionAttempt(BaseModel):
    """Traceable outcome for one ranked compression candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str
    target_tokens: int = Field(gt=0)
    attempted: bool
    result: CompressionResult | None = None
    reason: str | None = None


class CompressionExecution(BaseModel):
    """Complete compression-stage result and accounting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attempts: list[CompressionAttempt]
    successful_results: dict[str, CompressionResult]
    used_tokens: int = Field(ge=0)
    returned_tokens: int = Field(ge=0)


class CompressionExecutor:
    """Execute ranked candidates and deterministically reuse returned budget."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        extractive: Compressor | None = None,
        tool_output: Compressor | None = None,
        none: Compressor | None = None,
    ) -> None:
        self._extractive = extractive or ExtractiveCompressor(tokenizer)
        self._tool_output = tool_output or ToolOutputCompressor(tokenizer)
        self._none = none or NoneCompressor(tokenizer)

    def execute(
        self,
        plan: AllocationPlan,
        items: Sequence[ContextItem],
        *,
        task: str,
        policy: OptimizationPolicy,
    ) -> CompressionExecution:
        """Run every candidate that fits the budget available at its ranked turn."""
        by_id = {item.id: item for item in items}
        direct_by_type: dict[ContextType, int] = defaultdict(int)
        for selection in plan.direct_selected:
            direct_by_type[by_id[selection.item_id].type] += selection.allocated_tokens

        used_by_type = dict(direct_by_type)
        used_tokens = 0
        attempts: list[CompressionAttempt] = []
        successes: dict[str, CompressionResult] = {}
        for item_id in plan.compression_candidate_order:
            item = by_id[item_id]
            target = plan.compression_candidate_targets[item_id]
            available = plan.compression_budget - used_tokens
            maximum = policy.class_maximum_tokens.get(item.type)
            if target > available:
                attempts.append(
                    CompressionAttempt(
                        item_id=item_id,
                        target_tokens=target,
                        attempted=False,
                        reason="insufficient_compression_budget",
                    )
                )
                continue
            if maximum is not None and used_by_type.get(item.type, 0) + target > maximum:
                attempts.append(
                    CompressionAttempt(
                        item_id=item_id,
                        target_tokens=target,
                        attempted=False,
                        reason="class_maximum_exceeded",
                    )
                )
                continue
            compressor = self._compressor_for(item)
            try:
                result = compressor.compress(item, target, task)
            except Exception as exc:  # compressor boundary must not abort optimization
                attempts.append(
                    CompressionAttempt(
                        item_id=item_id,
                        target_tokens=target,
                        attempted=True,
                        reason=f"compressor_error:{type(exc).__name__}",
                    )
                )
                continue
            invalid_reason = self._validate_result(item, target, result)
            if invalid_reason is not None:
                attempts.append(
                    CompressionAttempt(
                        item_id=item_id,
                        target_tokens=target,
                        attempted=True,
                        result=result,
                        reason=invalid_reason,
                    )
                )
                continue
            successes[item_id] = result
            used_tokens += result.compressed_tokens
            used_by_type[item.type] = used_by_type.get(item.type, 0) + result.compressed_tokens
            attempts.append(
                CompressionAttempt(
                    item_id=item_id,
                    target_tokens=target,
                    attempted=True,
                    result=result,
                )
            )
        return CompressionExecution(
            attempts=attempts,
            successful_results=successes,
            used_tokens=used_tokens,
            returned_tokens=plan.compression_budget - used_tokens,
        )

    def _compressor_for(self, item: ContextItem) -> Compressor:
        if item.mandatory or not item.compressible or item.type in _PROTECTED_TYPES:
            return self._none
        if item.type in _TOOL_TYPES:
            return self._tool_output
        return self._extractive

    @staticmethod
    def _validate_result(
        item: ContextItem,
        target_tokens: int,
        result: CompressionResult,
    ) -> str | None:
        if not result.succeeded:
            return result.failure_reason or "compression_failed"
        if result.source_item_id != item.id or item.id not in result.provenance:
            return "invalid_provenance"
        if result.content is None or not result.content.strip():
            return "empty_result"
        if result.compressed_tokens > target_tokens:
            return "target_overflow"
        if result.compressed_tokens >= result.original_tokens:
            return "compression_not_beneficial"
        return None
