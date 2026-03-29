from __future__ import annotations

import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "storage" / "jatayu.sqlite3"
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def get_default_database_path() -> Path:
    return DEFAULT_DB_PATH


class Database:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else get_default_database_path()
        self._connection: sqlite3.Connection | None = None

    @property
    def connection(self) -> sqlite3.Connection:
        connection = self.connect()
        return connection

    def connect(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection

        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        self._connection = connection
        return connection

    def close(self) -> None:
        if self._connection is None:
            return

        self._connection.close()
        self._connection = None

    def migrate(self) -> list[str]:
        connection = self.connect()
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
            """
        )
        applied_versions = {
            row["version"]
            for row in connection.execute(
                "SELECT version FROM schema_migrations"
            ).fetchall()
        }

        applied_now: list[str] = []
        for migration_path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            version = migration_path.name
            if version in applied_versions:
                continue

            sql = migration_path.read_text(encoding="utf-8")
            with connection:
                connection.executescript(sql)
                connection.execute(
                    """
                    INSERT INTO schema_migrations (version, applied_at)
                    VALUES (?, datetime('now'))
                    """,
                    (version,),
                )
            applied_now.append(version)

        return applied_now
