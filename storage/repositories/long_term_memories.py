from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from storage.models import LongTermMemoryRecord


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _to_record(row: sqlite3.Row) -> LongTermMemoryRecord:
    return LongTermMemoryRecord(
        id=row["id"],
        scope_type=row["scope_type"],
        scope_key=row["scope_key"],
        channel_id=row["channel_id"],
        participant_id=row["participant_id"],
        memory_key=row["memory_key"],
        category=row["category"],
        summary=row["summary"],
        importance=row["importance"],
        confidence=row["confidence"],
        source_message_id=row["source_message_id"],
        status=row["status"],
        last_observed_at=_parse_datetime(row["last_observed_at"]),
        created_at=_parse_datetime(row["created_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
    )


class LongTermMemoryRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def upsert(
        self,
        *,
        scope_type: str,
        scope_key: str,
        channel_id: int | None,
        participant_id: int | None,
        memory_key: str,
        category: str,
        summary: str,
        importance: str,
        confidence: float,
        source_message_id: int | None,
        last_observed_at: datetime,
    ) -> LongTermMemoryRecord:
        now = _utcnow().isoformat()
        self.connection.execute(
            """
            INSERT INTO long_term_memories (
                scope_type,
                scope_key,
                channel_id,
                participant_id,
                memory_key,
                category,
                summary,
                importance,
                confidence,
                source_message_id,
                status,
                last_observed_at,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)
            ON CONFLICT(scope_type, scope_key, memory_key) DO UPDATE SET
                channel_id = excluded.channel_id,
                participant_id = excluded.participant_id,
                category = excluded.category,
                summary = excluded.summary,
                importance = excluded.importance,
                confidence = excluded.confidence,
                source_message_id = COALESCE(excluded.source_message_id, long_term_memories.source_message_id),
                status = 'active',
                last_observed_at = excluded.last_observed_at,
                updated_at = excluded.updated_at
            """,
            (
                scope_type,
                scope_key,
                channel_id,
                participant_id,
                memory_key,
                category,
                summary,
                importance,
                confidence,
                source_message_id,
                last_observed_at.isoformat(),
                now,
                now,
            ),
        )
        record = self.get_by_scope_and_key(
            scope_type=scope_type,
            scope_key=scope_key,
            memory_key=memory_key,
        )
        if record is None:
            raise RuntimeError("Failed to load upserted long-term memory")

        return record

    def archive(
        self,
        *,
        scope_type: str,
        scope_key: str,
        memory_key: str,
    ) -> LongTermMemoryRecord | None:
        now = _utcnow().isoformat()
        self.connection.execute(
            """
            UPDATE long_term_memories
            SET status = 'archived',
                updated_at = ?
            WHERE scope_type = ?
              AND scope_key = ?
              AND memory_key = ?
            """,
            (
                now,
                scope_type,
                scope_key,
                memory_key,
            ),
        )
        return self.get_by_scope_and_key(
            scope_type=scope_type,
            scope_key=scope_key,
            memory_key=memory_key,
        )

    def get_by_scope_and_key(
        self,
        *,
        scope_type: str,
        scope_key: str,
        memory_key: str,
    ) -> LongTermMemoryRecord | None:
        row = self.connection.execute(
            """
            SELECT * FROM long_term_memories
            WHERE scope_type = ?
              AND scope_key = ?
              AND memory_key = ?
            """,
            (
                scope_type,
                scope_key,
                memory_key,
            ),
        ).fetchone()
        if row is None:
            return None

        return _to_record(row)

    def list_relevant(
        self,
        *,
        channel_id: int,
        participant_id: int | None,
        limit: int,
    ) -> list[LongTermMemoryRecord]:
        rows = self.connection.execute(
            """
            SELECT *
            FROM long_term_memories
            WHERE status = 'active'
              AND (
                    (scope_type = 'channel' AND channel_id = ?)
                 OR (scope_type = 'participant' AND participant_id = ?)
              )
            ORDER BY
                CASE importance
                    WHEN 'high' THEN 3
                    WHEN 'medium' THEN 2
                    ELSE 1
                END DESC,
                updated_at DESC,
                id DESC
            LIMIT ?
            """,
            (
                channel_id,
                participant_id,
                limit,
            ),
        ).fetchall()
        return [_to_record(row) for row in rows]

    def list_all_active(self) -> list[LongTermMemoryRecord]:
        """Get all active long-term memories for compression purposes."""
        rows = self.connection.execute(
            """
            SELECT *
            FROM long_term_memories
            WHERE status = 'active'
            ORDER BY updated_at ASC
            """
        ).fetchall()
        return [_to_record(row) for row in rows]
