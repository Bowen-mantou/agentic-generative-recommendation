"""SQLite store for :class:`InteractionEvent` (TIGER behavior sequences).

The store is deliberately separate from Mem0 semantic memory (AGENTS.md §2.2, Q7).
Rows are append-only product events sorted by timestamp; ``event_id`` is a deterministic
hash of ``(user_id, product_id, timestamp, asin)`` so re-runs are idempotent.

Reviews are *proxy* signals: event types are only ``review`` / ``rating`` for Amazon
data. Click / purchase events can only be written by a first-party source later.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recommender.observability.logging import get_logger
from recommender.schemas.common import EventType
from recommender.schemas.memory import InteractionEvent
from recommender.utils.io import ensure_dir, utc_now_iso

_LOGGER = get_logger(__name__)

SCHEMA_VERSION = "1"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS interaction_events (
    event_id     TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    product_id   TEXT NOT NULL,
    event_type   TEXT NOT NULL CHECK (event_type IN (
        'review', 'rating', 'click', 'favorite', 'purchase',
        'explicit_positive_feedback',
        'explicit_negative_feedback'
    )),
    rating       INTEGER CHECK (rating IS NULL OR (rating >= 1 AND rating <= 5)),
    timestamp    INTEGER NOT NULL CHECK (timestamp >= 0),
    source       TEXT NOT NULL,
    source_id    TEXT,
    is_synthetic INTEGER NOT NULL CHECK (is_synthetic IN (0, 1)),
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_user_time ON interaction_events(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_product ON interaction_events(product_id);
CREATE INDEX IF NOT EXISTS idx_events_time ON interaction_events(timestamp);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

_INSERT_SQL = """
INSERT OR IGNORE INTO interaction_events
    (event_id, user_id, product_id, event_type, rating, timestamp, source, source_id,
     is_synthetic, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


@dataclass(frozen=True)
class InsertStats:
    attempted: int
    inserted: int
    duplicates: int

    def as_dict(self) -> dict[str, int]:
        return {
            "attempted": self.attempted,
            "inserted": self.inserted,
            "duplicates": self.duplicates,
        }


class InteractionStore:
    """Thin, typed wrapper around the `interaction_events` table."""

    def __init__(self, db_path: Path | str, *, create: bool = True) -> None:
        self.db_path = Path(db_path)
        if create:
            ensure_dir(self.db_path.parent)
        self._connection = sqlite3.connect(str(self.db_path))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.executescript(_SCHEMA)
        self._connection.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> InteractionStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- writes ---------------------------------------------------------------
    def insert_events(self, events: Sequence[InteractionEvent]) -> InsertStats:
        """Insert events idempotently (duplicate ``event_id`` is ignored)."""
        if not events:
            return InsertStats(attempted=0, inserted=0, duplicates=0)
        created_at = utc_now_iso()
        rows = [
            (
                event.event_id,
                event.user_id,
                event.product_id,
                event.event_type.value,
                event.rating,
                event.timestamp,
                event.source,
                event.source_id,
                1 if event.is_synthetic else 0,
                created_at,
            )
            for event in events
        ]
        before = self._connection.total_changes
        with self._connection:
            self._connection.executemany(_INSERT_SQL, rows)
        inserted = self._connection.total_changes - before
        return InsertStats(attempted=len(rows), inserted=inserted, duplicates=len(rows) - inserted)

    # -- reads ----------------------------------------------------------------
    def count(self, *, user_id: str | None = None, cutoff_ts: int | None = None) -> int:
        query = "SELECT COUNT(*) FROM interaction_events WHERE 1=1"
        params: list[Any] = []
        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)
        if cutoff_ts is not None:
            query += " AND timestamp <= ?"
            params.append(cutoff_ts)
        row = self._connection.execute(query, params).fetchone()
        return int(row[0])

    def events_for_user(
        self, user_id: str, *, cutoff_ts: int | None = None, limit: int | None = None
    ) -> list[InteractionEvent]:
        query = "SELECT * FROM interaction_events WHERE user_id = ?"
        params: list[Any] = [user_id]
        if cutoff_ts is not None:
            query += " AND timestamp <= ?"
            params.append(cutoff_ts)
        query += " ORDER BY timestamp ASC, event_id ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        return [self._row_to_event(row) for row in self._connection.execute(query, params)]

    def iter_user_sequences(
        self,
        *,
        min_events: int = 2,
        cutoff_ts: int | None = None,
    ) -> Iterator[tuple[str, list[InteractionEvent]]]:
        """Yield ``(user_id, chronologically ordered events)`` for users with enough data."""
        query = "SELECT * FROM interaction_events"
        params: list[Any] = []
        if cutoff_ts is not None:
            query += " WHERE timestamp <= ?"
            params.append(cutoff_ts)
        query += " ORDER BY user_id ASC, timestamp ASC, event_id ASC"
        current_user: str | None = None
        buffer: list[InteractionEvent] = []
        for row in self._connection.execute(query, params):
            event = self._row_to_event(row)
            if current_user is None:
                current_user = event.user_id
            if event.user_id != current_user:
                if len(buffer) >= min_events:
                    yield current_user, buffer
                current_user = event.user_id
                buffer = []
            buffer.append(event)
        if current_user is not None and len(buffer) >= min_events:
            yield current_user, buffer

    def sequence_lengths(self, *, cutoff_ts: int | None = None) -> list[int]:
        return [
            len(events) for _, events in self.iter_user_sequences(min_events=1, cutoff_ts=cutoff_ts)
        ]

    def stats(self, *, cutoff_ts: int | None = None) -> dict[str, Any]:
        total = self.count(cutoff_ts=cutoff_ts)
        users = self._connection.execute(
            "SELECT COUNT(DISTINCT user_id) FROM interaction_events"
            + (" WHERE timestamp <= ?" if cutoff_ts is not None else ""),
            ([cutoff_ts] if cutoff_ts is not None else []),
        ).fetchone()[0]
        products = self._connection.execute(
            "SELECT COUNT(DISTINCT product_id) FROM interaction_events"
            + (" WHERE timestamp <= ?" if cutoff_ts is not None else ""),
            ([cutoff_ts] if cutoff_ts is not None else []),
        ).fetchone()[0]
        event_types = {
            row[0]: row[1]
            for row in self._connection.execute(
                "SELECT event_type, COUNT(*) FROM interaction_events GROUP BY event_type"
            )
        }
        return {
            "db_path": str(self.db_path),
            "schema_version": SCHEMA_VERSION,
            "events": int(total),
            "users": int(users),
            "products": int(products),
            "event_types": event_types,
        }

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> InteractionEvent:
        event_type_value = row["event_type"]
        return InteractionEvent(
            event_id=row["event_id"],
            user_id=row["user_id"],
            product_id=row["product_id"],
            event_type=EventType(event_type_value),
            rating=row["rating"],
            timestamp=row["timestamp"],
            source=row["source"],
            source_id=row["source_id"],
            is_synthetic=bool(row["is_synthetic"]),
        )
