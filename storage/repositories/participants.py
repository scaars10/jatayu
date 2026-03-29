from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from storage.models import ParticipantRecord


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _to_record(row: sqlite3.Row) -> ParticipantRecord:
    return ParticipantRecord(
        id=row["id"],
        provider=row["provider"],
        external_id=row["external_id"],
        display_name=row["display_name"],
        metadata_json=row["metadata_json"],
        created_at=_parse_datetime(row["created_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
    )


class ParticipantRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def upsert(
        self,
        *,
        provider: str,
        external_id: str,
        display_name: str | None = None,
        metadata_json: str | None = None,
    ) -> ParticipantRecord:
        now = _utcnow().isoformat()
        self.connection.execute(
            """
            INSERT INTO participants (
                provider,
                external_id,
                display_name,
                metadata_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, external_id) DO UPDATE SET
                display_name = COALESCE(excluded.display_name, participants.display_name),
                metadata_json = COALESCE(excluded.metadata_json, participants.metadata_json),
                updated_at = excluded.updated_at
            """,
            (
                provider,
                external_id,
                display_name,
                metadata_json,
                now,
                now,
            ),
        )
        row = self.connection.execute(
            """
            SELECT * FROM participants
            WHERE provider = ? AND external_id = ?
            """,
            (provider, external_id),
        ).fetchone()
        if row is None:
            raise RuntimeError("Failed to load upserted participant")

        return _to_record(row)

    def get_by_provider_external_id(
        self,
        *,
        provider: str,
        external_id: str,
    ) -> ParticipantRecord | None:
        row = self.connection.execute(
            """
            SELECT * FROM participants
            WHERE provider = ? AND external_id = ?
            """,
            (provider, external_id),
        ).fetchone()
        if row is None:
            return None

        return _to_record(row)
