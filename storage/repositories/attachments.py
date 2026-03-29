from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from storage.models import MessageAttachmentRecord


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _to_record(row: sqlite3.Row) -> MessageAttachmentRecord:
    return MessageAttachmentRecord(
        id=row["id"],
        message_id=row["message_id"],
        media_type=row["media_type"],
        provider_file_id=row["provider_file_id"],
        provider_file_unique_id=row["provider_file_unique_id"],
        mime_type=row["mime_type"],
        file_name=row["file_name"],
        width=row["width"],
        height=row["height"],
        duration_seconds=row["duration_seconds"],
        file_size=row["file_size"],
        performer=row["performer"],
        title=row["title"],
        caption=row["caption"],
        media_group_id=row["media_group_id"],
        payload_json=row["payload_json"],
        created_at=_parse_datetime(row["created_at"]),
    )


class AttachmentRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create(
        self,
        *,
        message_id: int,
        media_type: str,
        provider_file_id: str,
        provider_file_unique_id: str,
        mime_type: str | None,
        file_name: str | None,
        width: int | None,
        height: int | None,
        duration_seconds: int | None,
        file_size: int | None,
        performer: str | None,
        title: str | None,
        caption: str | None,
        media_group_id: str | None,
        payload_json: str | None,
    ) -> MessageAttachmentRecord:
        created_at = _utcnow().isoformat()
        cursor = self.connection.execute(
            """
            INSERT INTO message_attachments (
                message_id,
                media_type,
                provider_file_id,
                provider_file_unique_id,
                mime_type,
                file_name,
                width,
                height,
                duration_seconds,
                file_size,
                performer,
                title,
                caption,
                media_group_id,
                payload_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                media_type,
                provider_file_id,
                provider_file_unique_id,
                mime_type,
                file_name,
                width,
                height,
                duration_seconds,
                file_size,
                performer,
                title,
                caption,
                media_group_id,
                payload_json,
                created_at,
            ),
        )
        row = self.connection.execute(
            "SELECT * FROM message_attachments WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()
        if row is None:
            raise RuntimeError("Failed to load created message attachment")

        return _to_record(row)
