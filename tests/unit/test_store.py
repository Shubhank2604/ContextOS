"""Tests for the in-memory context store."""

import pytest

from contextos.errors import ContextItemNotFoundError
from contextos.models import ContextItem, ContextType, LifecycleTier
from contextos.store import InMemoryContextStore


def test_save_and_load_returns_isolated_copy(context_item: ContextItem) -> None:
    store = InMemoryContextStore()
    store.save_item(context_item)

    context_item.content = "Caller mutation"
    loaded = store.load_item("item-1")
    assert loaded.content == "A useful context fact."

    loaded.content = "Loaded copy mutation"
    assert store.load_item("item-1").content == "A useful context fact."


def test_save_replaces_same_id(context_item: ContextItem) -> None:
    store = InMemoryContextStore()
    store.save_item(context_item)
    context_item.content = "Updated"
    store.save_item(context_item)
    assert store.load_item("item-1").content == "Updated"


def test_list_items_is_filtered_and_sorted(context_item: ContextItem) -> None:
    store = InMemoryContextStore()
    other = context_item.model_copy(
        update={
            "id": "a-item",
            "type": ContextType.ERROR,
            "lifecycle_tier": LifecycleTier.WARM,
        }
    )
    store.save_item(context_item)
    store.save_item(other)

    assert [item.id for item in store.list_items()] == ["a-item", "item-1"]
    assert store.list_items(context_type=ContextType.ERROR)[0].id == "a-item"
    assert store.list_items(lifecycle_tier=LifecycleTier.WARM)[0].id == "a-item"


def test_missing_item_operations_raise_typed_error() -> None:
    store = InMemoryContextStore()
    with pytest.raises(ContextItemNotFoundError):
        store.load_item("missing")
    with pytest.raises(ContextItemNotFoundError):
        store.delete_item("missing")


def test_delete_is_explicit(context_item: ContextItem) -> None:
    store = InMemoryContextStore()
    store.save_item(context_item)
    store.delete_item(context_item.id)
    assert store.list_items() == []
