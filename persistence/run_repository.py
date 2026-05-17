from __future__ import annotations

import json
import sqlite3
from typing import Any

from ber_tokenscope.schemas import RunRecord
from ber_tokenscope.settings import get_settings
from persistence.database import connect_sqlite


class RunRepository:
    """SQLite-backed run metadata repository."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or get_settings().database.url
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return connect_sqlite(self.database_url)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    artifact_path TEXT NOT NULL,
                    summary_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)"
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(task)")

    def save(self, record: RunRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id,
                    task,
                    status,
                    created_at,
                    duration_ms,
                    artifact_path,
                    summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.task,
                    record.status,
                    record.created_at,
                    record.duration_ms,
                    record.artifact_path,
                    json.dumps(record.summary),
                ),
            )

    def list(self, limit: int = 20, offset: int = 0) -> list[RunRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, task, status, created_at, duration_ms, artifact_path, summary_json
                FROM runs
                ORDER BY created_at DESC
                LIMIT ?
                OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [record_from_row(row) for row in rows]

    def get(self, run_id: str) -> RunRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT run_id, task, status, created_at, duration_ms, artifact_path, summary_json
                FROM runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        return None if row is None else record_from_row(row)

    def trim(self, max_history: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM runs
                WHERE run_id NOT IN (
                    SELECT run_id
                    FROM runs
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                """,
                (max_history,),
            )


def record_from_row(row: sqlite3.Row) -> RunRecord:
    summary: dict[str, Any] = json.loads(row["summary_json"])
    return RunRecord(
        run_id=row["run_id"],
        task=row["task"],
        status=row["status"],
        created_at=row["created_at"],
        duration_ms=row["duration_ms"],
        artifact_path=row["artifact_path"],
        summary=summary,
    )
