"""Tests for safe compression and deterministic budget reuse."""

from datetime import UTC, datetime

from contextos.budget import AllocationPlan, CompressionRequest
from contextos.compression import (
    CompressionExecutor,
    CompressionResult,
    ExtractiveCompressor,
    LLMSummaryCompressor,
    NoneCompressor,
    ToolOutputCompressor,
)
from contextos.config import OptimizationPolicy
from contextos.models import ContextItem, ContextType
from contextos.providers import MockLLMProvider
from contextos.providers.base import ProviderResponse


class WordTokenizer:
    """Small predictable tokenizer for compression contract tests."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def item(
    item_id: str,
    content: str,
    *,
    context_type: ContextType = ContextType.RETRIEVED_DOCUMENT,
) -> ContextItem:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return ContextItem(
        id=item_id,
        content=content,
        type=context_type,
        created_at=timestamp,
        updated_at=timestamp,
        token_count=len(content.split()),
    )


def test_extractive_preserves_exact_sentences_numbers_and_order() -> None:
    source = item(
        "doc",
        "Introductory filler is not useful. Timeout is exactly 37 seconds. Retry twice.",
    )
    result = ExtractiveCompressor(WordTokenizer()).compress(source, 7, "What is the timeout?")

    assert result.succeeded
    assert result.content is not None
    assert "Timeout is exactly 37 seconds." in result.content
    assert "37" in result.content
    assert result.provenance == ("doc",)
    assert result.compressed_tokens <= 7


def test_tool_output_keeps_error_line_and_does_not_rewrite_values() -> None:
    source = item(
        "log",
        "start request\nnoise noise noise\nERROR status=503 request_id=ABC-42\nend request",
        context_type=ContextType.TOOL_OUTPUT,
    )
    result = ToolOutputCompressor(WordTokenizer()).compress(source, 5, "request failed")

    assert result.succeeded
    assert result.content is not None
    assert "ERROR status=503 request_id=ABC-42" in result.content


def test_llm_summary_is_disabled_and_protects_code_and_identifiers() -> None:
    compressor = LLMSummaryCompressor(
        WordTokenizer(), MockLLMProvider("short summary"), enabled=True
    )
    code = item("code", "value = 123", context_type=ContextType.CODE)
    identifier = item("identifier", "Use request_id ABC-42 for lookup.")

    assert compressor.compress(code, 2, "task").failure_reason == "context_type_protected"
    assert (
        compressor.compress(identifier, 2, "task").failure_reason
        == "identifiers_or_secrets_protected"
    )


class UnavailableLLMProvider:
    def complete(self, prompt: str, *, max_output_tokens: int) -> ProviderResponse:
        del prompt, max_output_tokens
        raise RuntimeError("offline")


def test_optional_llm_unavailability_becomes_a_failed_attempt() -> None:
    source = item("prose", "ordinary prose has several useful words")
    compressor = LLMSummaryCompressor(WordTokenizer(), UnavailableLLMProvider(), enabled=True)
    assert (
        compressor.compress(source, 3, "summarize prose").failure_reason == "provider_unavailable"
    )


def test_empty_llm_result_and_none_overflow_are_explicit_failures() -> None:
    source = item("prose", "ordinary prose has several useful words")
    empty = LLMSummaryCompressor(WordTokenizer(), MockLLMProvider("   "), enabled=True)

    assert empty.compress(source, 3, "summarize").failure_reason == "empty_provider_result"
    assert (
        NoneCompressor(WordTokenizer()).compress(source, 2, "summarize").failure_reason
        == "content_exceeds_target"
    )


class FailingThenSuccessfulCompressor:
    """Fail the first reservation and compress the later candidate."""

    def compress(self, source: ContextItem, target_tokens: int, task: str) -> CompressionResult:
        del task
        if source.id == "first":
            return CompressionResult(
                original_tokens=source.token_count or 0,
                compressed_tokens=0,
                source_item_id=source.id,
                strategy="test",
                provenance=(source.id,),
                failure_reason="intentional_failure",
            )
        return CompressionResult(
            content="kept exact value",
            original_tokens=source.token_count or 0,
            compressed_tokens=3,
            source_item_id=source.id,
            strategy="test",
            provenance=(source.id,),
            lossy=True,
        )


def test_failed_reservation_is_reused_by_later_ranked_candidate() -> None:
    first = item("first", "one two three four five six seven eight nine ten")
    second = item("second", "one two three four five six seven eight nine ten")
    plan = AllocationPlan(
        direct_selected=[],
        compression_requests=[
            CompressionRequest(
                item_id="first",
                target_tokens=5,
                original_tokens=10,
                utility=0.9,
                value_density=0.18,
                reason="raw_item_did_not_fit",
            )
        ],
        rejected_item_ids=["second"],
        optional_budget=5,
        direct_tokens=0,
        compression_budget=5,
        candidate_item_ids=["first", "second"],
        rejection_reasons={"second": "insufficient_compression_budget"},
        class_allocated_tokens={ContextType.RETRIEVED_DOCUMENT: 5},
        compression_candidate_order=["first", "second"],
        compression_candidate_targets={"first": 5, "second": 5},
    )
    compressor = FailingThenSuccessfulCompressor()
    execution = CompressionExecutor(
        WordTokenizer(), extractive=compressor, tool_output=compressor, none=compressor
    ).execute(
        plan,
        [first, second],
        task="task",
        policy=OptimizationPolicy(max_input_tokens=5, minimum_compressed_tokens=1),
    )

    assert list(execution.successful_results) == ["second"]
    assert execution.used_tokens == 3
    assert execution.returned_tokens == 2
    assert execution.attempts[0].reason == "intentional_failure"
    assert execution.attempts[1].reason is None


class OverflowingCompressor:
    def compress(self, source: ContextItem, target_tokens: int, task: str) -> CompressionResult:
        del target_tokens, task
        return CompressionResult(
            content="one two three four six",
            original_tokens=source.token_count or 0,
            compressed_tokens=5,
            source_item_id=source.id,
            strategy="test",
            provenance=(source.id,),
        )


def test_executor_rejects_target_overflow() -> None:
    source = item("source", "one two three four five six seven eight")
    plan = AllocationPlan(
        direct_selected=[],
        compression_requests=[
            CompressionRequest(
                item_id="source",
                target_tokens=4,
                original_tokens=8,
                utility=0.8,
                value_density=0.2,
                reason="raw_item_did_not_fit",
            )
        ],
        rejected_item_ids=[],
        optional_budget=4,
        direct_tokens=0,
        compression_budget=4,
        candidate_item_ids=["source"],
        rejection_reasons={},
        compression_candidate_order=["source"],
        compression_candidate_targets={"source": 4},
    )
    compressor = OverflowingCompressor()
    execution = CompressionExecutor(WordTokenizer(), extractive=compressor).execute(
        plan,
        [source],
        task="task",
        policy=OptimizationPolicy(max_input_tokens=4, minimum_compressed_tokens=1),
    )

    assert not execution.successful_results
    assert execution.attempts[0].reason == "target_overflow"
