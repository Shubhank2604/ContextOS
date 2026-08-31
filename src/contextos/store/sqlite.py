"""Versioned SQLite persistence for context items and edges."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import cast

from contextos.errors import (
    ContextItemNotFoundError,
    CorruptedStoreError,
    StoreMigrationError,
    UnknownDependencyReference,
)
from contextos.models import ContextEdge, ContextItem, ContextType, LifecycleTier

CURRENT_SCHEMA_VERSION = 1


class SQLiteContextStore:
    """Durable local store with explicit schema migration."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        try:
            self._connection = sqlite3.connect(self.path)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._migrate()
        except sqlite3.DatabaseError as exc:
            if hasattr(self, "_connection"):
                self._connection.close()
            raise CorruptedStoreError(f"Unable to open SQLite context store: {exc}") from exc
        except Exception:
            if hasattr(self, "_connection"):
                self._connection.close()
            raise

    def _migrate(self) -> None:
        cursor = self._connection.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        row = cursor.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        version = int(row[0]) if row is not None else 0
        if version > CURRENT_SCHEMA_VERSION:
            raise StoreMigrationError(
                f"Store schema version {version} is newer than supported version "
                f"{CURRENT_SCHEMA_VERSION}"
            )
        if version < 1:
            cursor.executescript(
                """
                CREATE TABLE IF NOT EXISTS context_items (
                    id TEXT PRIMARY KEY,
                    context_type TEXT NOT NULL,
                    lifecycle_tier TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS context_edges (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    weight REAL NOT NULL,
                    PRIMARY KEY (source_id, target_id, relation),
                    FOREIGN KEY (source_id) REFERENCES context_items(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES context_items(id) ON DELETE CASCADE
                );
                """
            )
            cursor.execute("DELETE FROM schema_version")
            cursor.execute("INSERT INTO schema_version(version) VALUES (1)")
        self._connection.commit()

    def save_item(self, item: ContextItem) -> None:
        """Create or replace a serialized validated item."""
        self._execute(
            """
            INSERT INTO context_items(id, context_type, lifecycle_tier, updated_at, payload_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                context_type=excluded.context_type,
                lifecycle_tier=excluded.lifecycle_tier,
                updated_at=excluded.updated_at,
                payload_json=excluded.payload_json
            """,
            (
                item.id,
                item.type.value,
                item.lifecycle_tier.value,
                item.updated_at.isoformat(),
                item.model_dump_json(),
            ),
        )

    def load_item(self, item_id: str) -> ContextItem:
        """Load and validate one serialized item."""
        row = self._query_one("SELECT payload_json FROM context_items WHERE id = ?", (item_id,))
        if row is None:
            raise ContextItemNotFoundError(f"Context item {item_id!r} was not found")
        try:
            return ContextItem.model_validate_json(row[0])
        except (ValueError, TypeError) as exc:
            raise CorruptedStoreError(f"Stored context item {item_id!r} is invalid") from exc

    def list_items(
        self,
        *,
        lifecycle_tier: LifecycleTier | None = None,
        context_type: ContextType | None = None,
        updated_after: datetime | None = None,
        updated_before: datetime | None = None,
    ) -> list[ContextItem]:
        """Load items matching typed and time filters in deterministic order."""
        clauses: list[str] = []
        parameters: list[str] = []
        if lifecycle_tier is not None:
            clauses.append("lifecycle_tier = ?")
            parameters.append(lifecycle_tier.value)
        if context_type is not None:
            clauses.append("context_type = ?")
            parameters.append(context_type.value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._query_all(
            f"SELECT payload_json FROM context_items{where} ORDER BY id",
            tuple(parameters),
        )
        try:
            items = [ContextItem.model_validate_json(row[0]) for row in rows]
        except (ValueError, TypeError) as exc:
            raise CorruptedStoreError("Stored context item data is invalid") from exc
        return [
            item
            for item in items
            if (updated_after is None or item.updated_at >= updated_after)
            and (updated_before is None or item.updated_at <= updated_before)
        ]

    def delete_item(self, item_id: str) -> None:
        """Explicitly delete an item and its incident dependency edges."""
        cursor = self._execute("DELETE FROM context_items WHERE id = ?", (item_id,))
        if cursor.rowcount == 0:
            raise ContextItemNotFoundError(f"Context item {item_id!r} was not found")

    def save_edge(self, edge: ContextEdge) -> None:
        """Persist an edge only when both endpoint records exist."""
        placeholders = (edge.source_id, edge.target_id)
        rows = self._query_all("SELECT id FROM context_items WHERE id IN (?, ?)", placeholders)
        known = {str(row[0]) for row in rows}
        unknown = sorted(set(placeholders) - known)
        if unknown:
            raise UnknownDependencyReference(
                f"Unknown dependency endpoint IDs: {', '.join(unknown)}"
            )
        self._execute(
            """
            INSERT INTO context_edges(source_id, target_id, relation, weight)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(source_id, target_id, relation) DO UPDATE SET weight=excluded.weight
            """,
            (edge.source_id, edge.target_id, edge.relation.value, edge.weight),
        )

    def load_dependencies(self, item_id: str | None = None) -> list[ContextEdge]:
        """Load all or incident dependency edges deterministically."""
        query = "SELECT source_id, target_id, relation, weight FROM context_edges"
        parameters: tuple[str, ...] = ()
        if item_id is not None:
            query += " WHERE source_id = ? OR target_id = ?"
            parameters = (item_id, item_id)
        query += " ORDER BY source_id, target_id, relation"
        return [
            ContextEdge(source_id=row[0], target_id=row[1], relation=row[2], weight=row[3])
            for row in self._query_all(query, parameters)
        ]

    def update_tier(self, item_id: str, lifecycle_tier: LifecycleTier) -> None:
        """Update a validated item's lifecycle tier."""
        item = self.load_item(item_id)
        item.lifecycle_tier = lifecycle_tier
        self.save_item(item)

    def close(self) -> None:
        """Flush and close the underlying SQLite connection."""
        self._connection.close()

    def __enter__(self) -> SQLiteContextStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def _execute(self, query: str, parameters: tuple[object, ...]) -> sqlite3.Cursor:
        try:
            cursor = self._connection.execute(query, parameters)
            self._connection.commit()
            return cursor
        except sqlite3.DatabaseError as exc:
            raise CorruptedStoreError(f"SQLite write failed: {exc}") from exc

    def _query_one(self, query: str, parameters: tuple[object, ...]) -> sqlite3.Row | None:
        try:
            return cast(sqlite3.Row | None, self._connection.execute(query, parameters).fetchone())
        except sqlite3.DatabaseError as exc:
            raise CorruptedStoreError(f"SQLite read failed: {exc}") from exc

    def _query_all(self, query: str, parameters: tuple[object, ...]) -> list[sqlite3.Row]:
        try:
            return self._connection.execute(query, parameters).fetchall()
        except sqlite3.DatabaseError as exc:
            raise CorruptedStoreError(f"SQLite read failed: {exc}") from exc
