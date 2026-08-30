"""Tests for core context models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from contextos.errors import DuplicateContextItemError
from contextos.models import ContextItem, ContextType, validate_unique_item_ids


def make_item(**changes: object) -> ContextItem:
    values: dict[str, object] = {
        "id": "item-1",
        "content": "Keep this fact.",
        "type": ContextType.DECISION,
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "updated_at": datetime(2026, 1, 1, tzinfo=UTC),
        "importance": 0.8,
    }
    values.update(changes)
    return ContextItem.model_validate(values)


@pytest.mark.parametrize("importance", [-0.01, 1.01])
def test_importance_must_be_normalized(importance: float) -> None:
    with pytest.raises(ValidationError):
        make_item(importance=importance)


@pytest.mark.parametrize("field", ["id", "content"])
def test_blank_required_text_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        make_item(**{field: "  "})


def test_timestamps_must_be_timezone_aware() -> None:
    with pytest.raises(ValidationError):
        make_item(created_at=datetime(2026, 1, 1))


def test_mandatory_item_cannot_be_evictable() -> None:
    with pytest.raises(ValidationError):
        make_item(mandatory=True, evictable=True)


def test_valid_mandatory_item_is_accepted() -> None:
    item = make_item(mandatory=True, evictable=False)
    assert item.mandatory is True
    assert item.evictable is False


def test_content_assignment_invalidates_derived_fields() -> None:
    item = make_item(token_count=4, semantic_hash="abc")
    item.content = "A changed fact."
    assert item.token_count is None
    assert item.semantic_hash is None


def test_invalid_content_assignment_keeps_existing_content() -> None:
    item = make_item(token_count=4)
    with pytest.raises(ValidationError):
        item.content = ""
    assert item.content == "Keep this fact."
    assert item.token_count == 4


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make_item(unknown="value")


def test_failed_assignment_preserves_mandatory_invariant() -> None:
    item = make_item()
    with pytest.raises(ValueError, match="mandatory context cannot be evictable"):
        item.mandatory = True
    assert item.mandatory is False
    assert item.evictable is True


def test_failed_assignment_preserves_evictable_invariant() -> None:
    item = make_item(mandatory=True, evictable=False)
    with pytest.raises(ValueError, match="mandatory context cannot be evictable"):
        item.evictable = True
    assert item.mandatory is True
    assert item.evictable is False


def test_context_collection_ids_must_be_unique() -> None:
    with pytest.raises(DuplicateContextItemError, match="item-1"):
        validate_unique_item_ids([make_item(), make_item()])


def test_context_collection_accepts_unique_ids() -> None:
    validate_unique_item_ids([make_item(), make_item(id="item-2")])
