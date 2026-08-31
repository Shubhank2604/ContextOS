"""Shared store contract, SQLite durability, and lifecycle tests."""

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from contextos.errors import CorruptedStoreError, UnknownDependencyReference
from contextos.lifecycle import LifecycleManager, LifecyclePolicy
from contextos.models import (
    ContextEdge,
    ContextItem,
    ContextType,
    DependencyRelation,
    LifecycleTier,
)
from contextos.store import ContextStore, InMemoryContextStore, SQLiteContextStore


def make_item(
    item_id: str,
    *,
    updated_at: datetime,
    tier: LifecycleTier = LifecycleTier.HOT,
    context_type: ContextType = ContextType.MEMORY,
) -> ContextItem:
    return ContextItem(
        id=item_id,
        content=f"content for {item_id}",
        type=context_type,
        created_at=updated_at,
        updated_at=updated_at,
        lifecycle_tier=tier,
    )


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[ContextStore]:
    if request.param == "memory":
        yield InMemoryContextStore()
        return
    sqlite_store = SQLiteContextStore(tmp_path / "contract.db")
    yield sqlite_store
    sqlite_store.close()


def test_shared_store_contract_items_filters_edges_and_tiers(store: ContextStore) -> None:
    now = datetime(2026, 1, 2, tzinfo=UTC)
    older = make_item("a", updated_at=now - timedelta(days=1), tier=LifecycleTier.WARM)
    newer = make_item("b", updated_at=now, context_type=ContextType.ERROR)
    store.save_item(older)
    store.save_item(newer)
    edge = ContextEdge(
        source_id="b",
        target_id="a",
        relation=DependencyRelation.REQUIRES,
        weight=0.8,
    )
    store.save_edge(edge)

    assert [item.id for item in store.list_items()] == ["a", "b"]
    assert [item.id for item in store.list_items(updated_after=now)] == ["b"]
    assert store.list_items(context_type=ContextType.ERROR)[0].id == "b"
    assert store.load_dependencies("a") == [edge]
    store.update_tier("a", LifecycleTier.COLD)
    assert store.load_item("a").lifecycle_tier is LifecycleTier.COLD
    store.delete_item("a")
    assert store.load_dependencies() == []


def test_shared_store_rejects_unknown_edge_endpoints(store: ContextStore) -> None:
    edge = ContextEdge(
        source_id="missing-a",
        target_id="missing-b",
        relation=DependencyRelation.REQUIRES,
        weight=1.0,
    )
    with pytest.raises(UnknownDependencyReference):
        store.save_edge(edge)


def test_sqlite_write_restart_read(tmp_path: Path) -> None:
    path = tmp_path / "restart.db"
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    first = SQLiteContextStore(path)
    first.save_item(make_item("durable", updated_at=timestamp))
    first.close()

    second = SQLiteContextStore(path)
    assert second.load_item("durable").content == "content for durable"
    second.close()


def test_sqlite_migrates_version_zero_fixture(tmp_path: Path) -> None:
    path = tmp_path / "migration.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    connection.execute("INSERT INTO schema_version(version) VALUES (0)")
    connection.commit()
    connection.close()

    store = SQLiteContextStore(path)
    assert store._connection.execute("SELECT version FROM schema_version").fetchone()[0] == 1
    store.close()


def test_corrupted_sqlite_behavior_is_typed(tmp_path: Path) -> None:
    path = tmp_path / "corrupted.db"
    path.write_text("this is not sqlite", encoding="utf-8")
    with pytest.raises(CorruptedStoreError):
        SQLiteContextStore(path)


def test_lifecycle_is_deterministic_and_archived_records_are_retained() -> None:
    now = datetime(2026, 2, 1, tzinfo=UTC)
    store = InMemoryContextStore()
    warm = make_item("warm", updated_at=now - timedelta(days=2))
    archived = make_item(
        "archived",
        updated_at=now - timedelta(days=100),
        tier=LifecycleTier.ARCHIVED,
    )
    store.save_item(warm)
    store.save_item(archived)
    manager = LifecycleManager(
        LifecyclePolicy(hot_seconds=86_400, warm_seconds=604_800, cold_seconds=1_209_600)
    )

    assert manager.transition_store(store, now=now) == {"warm": LifecycleTier.WARM}
    assert store.load_item("archived").lifecycle_tier is LifecycleTier.ARCHIVED
    assert [item.id for item in store.list_items()] == ["archived", "warm"]
