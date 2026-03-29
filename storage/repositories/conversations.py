from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from storage.models import ConversationRecord


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _to_record(row: sqlite3.Row) -> ConversationRecord:
    return ConversationRecord(
        id=row["id"],
        channel_id=row["channel_id"],
        status=row["status"],
        summary=row["summary"],
        started_at=_parse_datetime(row["started_at"]),
        last_message_at=_parse_datetime(row["last_message_at"]),
        created_at=_parse_datetime(row["created_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
    )


class ConversationRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def get_or_create_active(
        self,
        *,
        channel_id: int,
        started_at: datetime,
    ) -> ConversationRecord:
        row = self.connection.execute(
            """
            SELECT * FROM conversations
            WHERE channel_id = ? AND status = 'active'
            """,
            (channel_id,),
        ).fetchone()
        if row is not None:
            return _to_record(row)

        started_at_value = started_at.isoformat()
        now = _utcnow().isoformat()
        cursor = self.connection.execute(
            """
            INSERT INTO conversations (
                channel_id,
                status,
                summary,
                started_at,
                last_message_at,
                created_at,
                updated_at
            )
            VALUES (?, 'active', NULL, ?, ?, ?, ?)
            """,
            (
                channel_id,
                started_at_value,
                started_at_value,
                now,
                now,
            ),
        )
        created_row = self.connection.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()
        if created_row is None:
            raise RuntimeError("Failed to create active conversation")

        return _to_record(created_row)

    def attach_participant(
        self,
        *,
        conversation_id: int,
        participant_id: int,
        joined_at: datetime,
    ) -> None:
        self.connection.execute(
            """
            INSERT OR IGNORE INTO conversation_participants (
                conversation_id,
                participant_id,
                joined_at
            )
            VALUES (?, ?, ?)
            """,
            (
                conversation_id,
                participant_id,
                joined_at.isoformat(),
            ),
        )

    def touch(
        self,
        *,
        conversation_id: int,
        last_message_at: datetime,
    ) -> None:
        now = _utcnow().isoformat()
        self.connection.execute(
            """
            UPDATE conversations
            SET last_message_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                last_message_at.isoformat(),
                now,
                conversation_id,
            ),
        )
