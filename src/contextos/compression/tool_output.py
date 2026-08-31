"""Structure-aware compression for logs and tool output."""

from __future__ import annotations

import re

from contextos.compression.base import CompressionResult
from contextos.models import ContextItem
from contextos.tokenization import Tokenizer

_CRITICAL = re.compile(
    r"(?:error|exception|failed|failure|warning|warn|status\s*[:=]?\s*[45]\d\d|"
    r"\b[45]\d\d\b|traceback)",
    re.IGNORECASE,
)
_TASK_TERM = re.compile(r"[A-Za-z0-9_.:/\\-]{3,}")


class ToolOutputCompressor:
    """Keep critical, task-related, and boundary lines exactly as observed."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer

    def compress(
        self,
        item: ContextItem,
        target_tokens: int,
        task: str,
    ) -> CompressionResult:
        """Select exact source lines and emit them in their original order."""
        original_tokens = self._tokenizer.count_tokens(item.content)
        lines = item.content.splitlines()
        if not lines:
            return self._failure(item, original_tokens, "no_lines")
        task_terms = {term.casefold() for term in _TASK_TERM.findall(task)}

        def priority(index: int) -> tuple[int, int]:
            line = lines[index]
            terms = {term.casefold() for term in _TASK_TERM.findall(line)}
            if _CRITICAL.search(line):
                return (0, index)
            if task_terms & terms:
                return (1, index)
            if index in {0, len(lines) - 1}:
                return (2, index)
            return (3, index)

        selected: set[int] = set()
        for index in sorted(range(len(lines)), key=priority):
            proposed = "\n".join(
                line
                for line_index, line in enumerate(lines)
                if line_index in selected or line_index == index
            )
            if proposed.strip() and self._tokenizer.count_tokens(proposed) <= target_tokens:
                selected.add(index)
        content = "\n".join(line for index, line in enumerate(lines) if index in selected)
        if not content.strip():
            return self._failure(item, original_tokens, "no_line_fits_target")
        return CompressionResult(
            content=content,
            original_tokens=original_tokens,
            compressed_tokens=self._tokenizer.count_tokens(content),
            source_item_id=item.id,
            strategy="tool_output",
            provenance=(item.id,),
            lossy=content != item.content,
        )

    @staticmethod
    def _failure(item: ContextItem, original_tokens: int, reason: str) -> CompressionResult:
        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=0,
            source_item_id=item.id,
            strategy="tool_output",
            provenance=(item.id,),
            failure_reason=reason,
        )
