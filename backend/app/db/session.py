from __future__ import annotations

import sqlite3
from typing import Generator

from backend.app.core.config import settings


def connect_to_sqlite() -> sqlite3.Connection:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        settings.database_path,
        timeout=30,
        detect_types=sqlite3.PARSE_DECLTYPES,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def get_db() -> Generator[sqlite3.Connection, None, None]:
    connection = connect_to_sqlite()
    try:
        yield connection
    finally:
        connection.close()
