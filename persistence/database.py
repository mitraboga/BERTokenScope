from __future__ import annotations

import sqlite3
from pathlib import Path


def sqlite_path_from_url(database_url: str) -> Path:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise ValueError("Only sqlite:/// URLs are supported by the local repository.")
    return Path(database_url.removeprefix(prefix))


def connect_sqlite(database_url: str) -> sqlite3.Connection:
    path = sqlite_path_from_url(database_url)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection
