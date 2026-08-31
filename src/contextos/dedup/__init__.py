"""Exact and semantic context deduplication."""

from contextos.dedup.base import DeduplicationResult, DuplicateMatch
from contextos.dedup.exact import exact_deduplicate, normalize_content, normalized_content_hash
from contextos.dedup.metrics import (
    DeduplicationCase,
    DeduplicationMetrics,
    evaluate_deduplication_cases,
)
from contextos.dedup.semantic import semantic_deduplicate

__all__ = [
    "DeduplicationCase",
    "DeduplicationMetrics",
    "DeduplicationResult",
    "DuplicateMatch",
    "evaluate_deduplication_cases",
    "exact_deduplicate",
    "normalize_content",
    "normalized_content_hash",
    "semantic_deduplicate",
]
