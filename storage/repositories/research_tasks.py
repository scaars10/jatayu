from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from storage.models import ResearchTaskRecord


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _to_record(row: sqlite3.Row) -> ResearchTaskRecord:
    return ResearchTaskRecord(
        id=row["id"],
        topic=row["topic"],
        specific_questions=row["specific_questions"],
        status=row["status"],
        step=row["step"],
        sources_content=row["sources_content"],
        report=row["report"],
        feedback=row["feedback"],
        created_at=_parse_datetime(row["created_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
    )


class ResearchTaskRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create(
        self,
        *,
        topic: str,
        specific_questions: str | None,
    ) -> ResearchTaskRecord:
        now = _utcnow().isoformat()
        cursor = self.connection.execute(
            """
            INSERT INTO research_tasks (
                topic,
                specific_questions,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                topic,
                specific_questions,
                now,
                now,
            ),
        )
        created_id = cursor.lastrowid
        if created_id is None:
            raise RuntimeError("Failed to get last inserted rowid")
            
        created = self.get_by_id(created_id)
        if created is None:
            raise RuntimeError("Failed to load created research task")

        self.connection.commit()
        return created

    def get_by_id(self, task_id: int) -> ResearchTaskRecord | None:
        row = self.connection.execute(
            "SELECT * FROM research_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return None

        return _to_record(row)

    def update_status(
        self,
        *,
        task_id: int,
        status: str,
    ) -> ResearchTaskRecord | None:
        now = _utcnow().isoformat()
        self.connection.execute(
            """
            UPDATE research_tasks
            SET status = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                status,
                now,
                task_id,
            ),
        )
        self.connection.commit()
        return self.get_by_id(task_id)

    def update_report(
        self,
        *,
        task_id: int,
        report: str,
    ) -> ResearchTaskRecord | None:
        now = _utcnow().isoformat()
        self.connection.execute(
            """
            UPDATE research_tasks
            SET report = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                report,
                now,
                task_id,
            ),
        )
        self.connection.commit()
        return self.get_by_id(task_id)

    def update_feedback(
        self,
        *,
        task_id: int,
        feedback: str,
    ) -> ResearchTaskRecord | None:
        now = _utcnow().isoformat()
        self.connection.execute(
            """
            UPDATE research_tasks
            SET feedback = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                feedback,
                now,
                task_id,
            ),
        )
        self.connection.commit()
        return self.get_by_id(task_id)

    def update_step(
        self,
        *,
        task_id: int,
        step: str,
    ) -> ResearchTaskRecord | None:
        now = _utcnow().isoformat()
        self.connection.execute(
            """
            UPDATE research_tasks
            SET step = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                step,
                now,
                task_id,
            ),
        )
        self.connection.commit()
        return self.get_by_id(task_id)

    def update_sources_content(
        self,
        *,
        task_id: int,
        sources_content: str,
    ) -> ResearchTaskRecord | None:
        now = _utcnow().isoformat()
        self.connection.execute(
            """
            UPDATE research_tasks
            SET sources_content = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                sources_content,
                now,
                task_id,
            ),
        )
        self.connection.commit()
        return self.get_by_id(task_id)
