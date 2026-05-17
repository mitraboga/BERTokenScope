from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

from ber_tokenscope.settings import get_settings
from persistence.database import connect_sqlite


class IdempotencyRepository:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or get_settings().database.url
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return connect_sqlite(self.database_url)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS idempotency_keys (
                    idempotency_key TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def get(self, key: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT idempotency_key, task, request_hash, response_json, created_at
                FROM idempotency_keys
                WHERE idempotency_key = ?
                """,
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "key": row["idempotency_key"],
            "task": row["task"],
            "request_hash": row["request_hash"],
            "response": json.loads(row["response_json"]),
            "created_at": row["created_at"],
        }

    def save(
        self,
        *,
        key: str,
        task: str,
        request_hash: str,
        response: dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO idempotency_keys (
                    idempotency_key,
                    task,
                    request_hash,
                    response_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    key,
                    task,
                    request_hash,
                    json.dumps(response),
                    datetime.now(UTC).isoformat(),
                ),
            )
