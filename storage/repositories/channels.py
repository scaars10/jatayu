from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from storage.models import ChannelRecord


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _to_record(row: sqlite3.Row) -> ChannelRecord:
    return ChannelRecord(
        id=row["id"],
        provider=row["provider"],
        external_id=row["external_id"],
        channel_type=row["channel_type"],
        title=row["title"],
        metadata_json=row["metadata_json"],
        created_at=_parse_datetime(row["created_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
    )


class ChannelRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def upsert(
        self,
        *,
        provider: str,
        external_id: str,
        channel_type: str | None = None,
        title: str | None = None,
        metadata_json: str | None = None,
    ) -> ChannelRecord:
        now = _utcnow().isoformat()
        self.connection.execute(
            """
            INSERT INTO channels (
                provider,
                external_id,
                channel_type,
                title,
                metadata_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, external_id) DO UPDATE SET
                channel_type = COALESCE(excluded.channel_type, channels.channel_type),
                title = COALESCE(excluded.title, channels.title),
                metadata_json = COALESCE(excluded.metadata_json, channels.metadata_json),
                updated_at = excluded.updated_at
            """,
            (
                provider,
                external_id,
                channel_type,
                title,
                metadata_json,
                now,
                now,
            ),
        )
        row = self.connection.execute(
            """
            SELECT * FROM channels
            WHERE provider = ? AND external_id = ?
            """,
            (provider, external_id),
        ).fetchone()
        if row is None:
            raise RuntimeError("Failed to load upserted channel")

        return _to_record(row)

    def get_by_provider_external_id(
        self,
        *,
        provider: str,
        external_id: str,
    ) -> ChannelRecord | None:
        row = self.connection.execute(
            """
            SELECT * FROM channels
            WHERE provider = ? AND external_id = ?
            """,
            (provider, external_id),
        ).fetchone()
        if row is None:
            return None

        return _to_record(row)
