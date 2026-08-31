"""No-op compression baseline."""

from contextos.compression.base import CompressionResult
from contextos.models import ContextItem
from contextos.tokenization import Tokenizer


class NoneCompressor:
    """Return content unchanged only when it already fits the target."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer

    def compress(
        self,
        item: ContextItem,
        target_tokens: int,
        task: str,
    ) -> CompressionResult:
        """Apply the explicit no-op baseline."""
        del task
        original_tokens = self._tokenizer.count_tokens(item.content)
        if original_tokens > target_tokens:
            return CompressionResult(
                original_tokens=original_tokens,
                compressed_tokens=0,
                source_item_id=item.id,
                strategy="none",
                provenance=(item.id,),
                failure_reason="content_exceeds_target",
            )
        return CompressionResult(
            content=item.content,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            source_item_id=item.id,
            strategy="none",
            provenance=(item.id,),
        )
