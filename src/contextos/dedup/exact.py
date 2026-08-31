"""Unicode-aware exact deduplication with mandatory precedence."""

from __future__ import annotations

import hashlib
import unicodedata
from collections import defaultdict
from collections.abc import Sequence

from contextos.dedup.base import DeduplicationResult, DuplicateMatch
from contextos.models import ContextItem, validate_unique_item_ids


def normalize_content(content: str) -> str:
    """Normalize Unicode, line endings, and surrounding whitespace."""
    normalized = unicodedata.normalize("NFC", content)
    return normalized.replace("\r\n", "\n").replace("\r", "\n").strip()


def normalized_content_hash(content: str) -> str:
    """Return the SHA-256 digest of normalized content."""
    return hashlib.sha256(normalize_content(content).encode("utf-8")).hexdigest()


def _optional_priority(item: ContextItem) -> tuple[float, float, float, str]:
    return (
        -item.importance,
        -item.updated_at.timestamp(),
        -item.created_at.timestamp(),
        item.id,
    )


def exact_deduplicate(items: Sequence[ContextItem]) -> DeduplicationResult:
    """Remove only optional exact duplicates according to canonical precedence."""
    validate_unique_item_ids(items)
    copied = [item.model_copy(deep=True) for item in items]
    groups: dict[str, list[ContextItem]] = defaultdict(list)
    for item in copied:
        content_hash = normalized_content_hash(item.content)
        item.semantic_hash = content_hash
        groups[content_hash].append(item)

    matches: list[DuplicateMatch] = []
    warnings: list[str] = []
    removed_ids: set[str] = set()
    for group in groups.values():
        if len(group) < 2:
            continue
        mandatory = sorted((item for item in group if item.mandatory), key=lambda item: item.id)
        optional = [item for item in group if not item.mandatory]
        if len(mandatory) > 1:
            warnings.append("mandatory_duplicate:" + ",".join(item.id for item in mandatory))
        if mandatory:
            canonical = mandatory[0]
            for item in optional:
                removed_ids.add(item.id)
                matches.append(
                    DuplicateMatch(
                        item_id=item.id,
                        duplicate_of=canonical.id,
                        reason="duplicate_of_mandatory",
                    )
                )
            continue
        canonical = sorted(optional, key=_optional_priority)[0]
        for item in optional:
            if item.id == canonical.id:
                continue
            removed_ids.add(item.id)
            matches.append(
                DuplicateMatch(
                    item_id=item.id,
                    duplicate_of=canonical.id,
                    reason="exact_duplicate",
                )
            )

    return DeduplicationResult(
        items=[item for item in copied if item.id not in removed_ids],
        matches=sorted(matches, key=lambda match: match.item_id),
        warnings=sorted(warnings),
    )
