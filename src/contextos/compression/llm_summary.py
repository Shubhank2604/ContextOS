"""Optional, explicitly lossy LLM summarization."""

from contextos.compression.base import CompressionResult
from contextos.dedup.safety import semantic_safety_differences
from contextos.models import ContextItem, ContextType
from contextos.providers import LLMProvider
from contextos.tokenization import Tokenizer

_PROTECTED_TYPES = {
    ContextType.SYSTEM_INSTRUCTION,
    ContextType.TOOL_DEFINITION,
    ContextType.CODE,
}


class LLMSummaryCompressor:
    """Summarize eligible prose through an injected provider."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        provider: LLMProvider,
        *,
        enabled: bool = False,
    ) -> None:
        self._tokenizer = tokenizer
        self._provider = provider
        self._enabled = enabled

    def compress(
        self,
        item: ContextItem,
        target_tokens: int,
        task: str,
    ) -> CompressionResult:
        """Generate a bounded summary only for explicitly eligible content."""
        original_tokens = self._tokenizer.count_tokens(item.content)
        reason = self._ineligible_reason(item)
        if reason is not None:
            return self._failure(item, original_tokens, reason)
        prompt = (
            "Summarize the source for the stated task. Preserve exact facts and do not add facts.\n"
            f"Task: {task}\nSource:\n{item.content}"
        )
        try:
            response = self._provider.complete(prompt, max_output_tokens=target_tokens)
        except Exception:
            return self._failure(item, original_tokens, "provider_unavailable")
        content = response.text.strip()
        if not content:
            return self._failure(item, original_tokens, "empty_provider_result")
        compressed_tokens = self._tokenizer.count_tokens(content)
        if compressed_tokens > target_tokens:
            return self._failure(item, original_tokens, "provider_result_exceeds_target")
        return CompressionResult(
            content=content,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            source_item_id=item.id,
            strategy="llm_summary",
            provenance=(item.id,),
            lossy=True,
        )

    def _ineligible_reason(self, item: ContextItem) -> str | None:
        if not self._enabled:
            return "llm_compression_disabled"
        if item.mandatory:
            return "mandatory_content_protected"
        if not item.compressible:
            return "item_not_compressible"
        if item.type in _PROTECTED_TYPES:
            return "context_type_protected"
        if semantic_safety_differences(item.content, "", context_type=item.type):
            return "identifiers_or_secrets_protected"
        return None

    @staticmethod
    def _failure(item: ContextItem, original_tokens: int, reason: str) -> CompressionResult:
        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=0,
            source_item_id=item.id,
            strategy="llm_summary",
            provenance=(item.id,),
            failure_reason=reason,
            lossy=True,
        )
