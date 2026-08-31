"""Tests for exact deduplication precedence and determinism."""

from datetime import UTC, datetime, timedelta

from contextos.dedup import exact_deduplicate, normalize_content, normalized_content_hash
from contextos.models import ContextItem, ContextType


def _item(
    item_id: str,
    content: str,
    *,
    mandatory: bool = False,
    importance: float = 0.5,
    created_offset: int = 0,
    updated_offset: int = 0,
) -> ContextItem:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return ContextItem(
        id=item_id,
        content=content,
        type=ContextType.DECISION,
        created_at=start + timedelta(seconds=created_offset),
        updated_at=start + timedelta(seconds=updated_offset),
        importance=importance,
        mandatory=mandatory,
        evictable=not mandatory,
    )


def test_normalization_handles_unicode_line_endings_and_outer_whitespace() -> None:
    assert normalize_content("  Café\r\n") == "Café"
    assert normalized_content_hash("Café") == normalized_content_hash("Café\r\n")


def test_mandatory_duplicates_all_survive_with_warning() -> None:
    result = exact_deduplicate(
        [_item("required-b", "same", mandatory=True), _item("required-a", "same", mandatory=True)]
    )
    assert [item.id for item in result.items] == ["required-b", "required-a"]
    assert result.matches == []
    assert result.warnings == ["mandatory_duplicate:required-a,required-b"]


def test_optional_duplicate_of_mandatory_uses_lexicographically_smallest_target() -> None:
    mandatory_z = _item("mandatory-z", "same", mandatory=True)
    optional = _item("optional", "same", importance=1.0)
    optional.type = ContextType.PLAN
    mandatory_a = _item("mandatory-a", "same", mandatory=True)
    result = exact_deduplicate([mandatory_z, optional, mandatory_a])
    assert [item.id for item in result.items] == ["mandatory-z", "mandatory-a"]
    assert result.matches_by_item_id["optional"].duplicate_of == "mandatory-a"
    assert result.matches_by_item_id["optional"].reason == "duplicate_of_mandatory"


def test_optional_representative_priority_is_deterministic() -> None:
    result = exact_deduplicate(
        [
            _item("older-high", "same", importance=0.9, updated_offset=1),
            _item("newer-low", "same", importance=0.8, updated_offset=100),
            _item("newer-high-z", "same", importance=0.9, updated_offset=2),
            _item("newer-high-a", "same", importance=0.9, updated_offset=2),
        ]
    )
    assert [item.id for item in result.items] == ["newer-high-a"]
    assert {match.duplicate_of for match in result.matches} == {"newer-high-a"}
    assert all(item.semantic_hash is not None for item in result.items)
