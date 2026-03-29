from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from storage.models import ConversationTurn, MessageRecord


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _to_record(row: sqlite3.Row) -> MessageRecord:
    return MessageRecord(
        id=row["id"],
        event_id=row["event_id"],
        conversation_id=row["conversation_id"],
        channel_id=row["channel_id"],
        participant_id=row["participant_id"],
        source=row["source"],
        direction=row["direction"],
        role=row["role"],
        message_type=row["message_type"],
        provider_message_id=row["provider_message_id"],
        reply_to_provider_message_id=row["reply_to_provider_message_id"],
        request_event_id=row["request_event_id"],
        text_content=row["text_content"],
        payload_json=row["payload_json"],
        delivery_status=row["delivery_status"],
        occurred_at=_parse_datetime(row["occurred_at"]),
        created_at=_parse_datetime(row["created_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
    )


class MessageRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create(
        self,
        *,
        event_id: str,
        conversation_id: int,
        channel_id: int,
        participant_id: int | None,
        source: str,
        direction: str,
        role: str,
        message_type: str,
        provider_message_id: str | None,
        reply_to_provider_message_id: str | None,
        request_event_id: str | None,
        text_content: str | None,
        payload_json: str,
        delivery_status: str,
        occurred_at: datetime,
    ) -> MessageRecord:
        now = _utcnow().isoformat()
        self.connection.execute(
            """
            INSERT INTO messages (
                event_id,
                conversation_id,
                channel_id,
                participant_id,
                source,
                direction,
                role,
                message_type,
                provider_message_id,
                reply_to_provider_message_id,
                request_event_id,
                text_content,
                payload_json,
                delivery_status,
                occurred_at,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                conversation_id,
                channel_id,
                participant_id,
                source,
                direction,
                role,
                message_type,
                provider_message_id,
                reply_to_provider_message_id,
                request_event_id,
                text_content,
                payload_json,
                delivery_status,
                occurred_at.isoformat(),
                now,
                now,
            ),
        )
        created = self.get_by_event_id(event_id)
        if created is None:
            raise RuntimeError("Failed to load created message")

        return created

    def get_by_event_id(self, event_id: str) -> MessageRecord | None:
        row = self.connection.execute(
            "SELECT * FROM messages WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        if row is None:
            return None

        return _to_record(row)

    def mark_delivered(
        self,
        *,
        event_id: str,
        provider_message_id: str,
    ) -> MessageRecord | None:
        current = self.get_by_event_id(event_id)
        if current is None:
            return None

        now = _utcnow().isoformat()
        self.connection.execute(
            """
            UPDATE messages
            SET provider_message_id = ?,
                delivery_status = 'delivered',
                updated_at = ?
            WHERE event_id = ?
            """,
            (
                provider_message_id,
                now,
                event_id,
            ),
        )
        return self.get_by_event_id(event_id)

    def list_recent_turns(
        self,
        *,
        channel_id: int,
        limit: int,
    ) -> list[ConversationTurn]:
        rows = self.connection.execute(
            """
            SELECT m.role, m.text_content, m.source, m.message_type, m.occurred_at
            FROM messages AS m
            INNER JOIN conversations AS c
                ON c.id = m.conversation_id
            WHERE c.channel_id = ?
              AND c.status = 'active'
              AND m.text_content IS NOT NULL
              AND TRIM(m.text_content) != ''
            ORDER BY m.occurred_at DESC, m.id DESC
            LIMIT ?
            """,
            (channel_id, limit),
        ).fetchall()
        turns = [
            ConversationTurn(
                role=row["role"],
                text=row["text_content"],
                source=row["source"],
                message_type=row["message_type"],
                occurred_at=_parse_datetime(row["occurred_at"]),
            )
            for row in rows
        ]
        turns.reverse()
        return turns
