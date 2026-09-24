"""SQLite persistence for scenes, trained models and processing jobs."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS scenes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT 'upload',
    sensor        TEXT NOT NULL DEFAULT 'reflectance',
    dos           INTEGER NOT NULL DEFAULT 0,
    folder        TEXT NOT NULL,
    has_dem       INTEGER NOT NULL DEFAULT 0,
    has_reference INTEGER NOT NULL DEFAULT 0,
    has_cloud     INTEGER NOT NULL DEFAULT 0,
    width         INTEGER,
    height        INTEGER,
    crs           TEXT,
    bounds        TEXT,
    pixel_size    REAL,
    created       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT NOT NULL,
    folder    TEXT NOT NULL,
    scene_id  INTEGER,
    uses_dem  INTEGER NOT NULL DEFAULT 0,
    best_algo TEXT,
    summary   TEXT,
    created   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    kind      TEXT NOT NULL,
    status    TEXT NOT NULL DEFAULT 'queued',
    progress  REAL NOT NULL DEFAULT 0,
    message   TEXT,
    params    TEXT,
    scene_id  INTEGER,
    model_id  INTEGER,
    folder    TEXT,
    error     TEXT,
    created   TEXT NOT NULL,
    finished  TEXT
);
"""

JSON_FIELDS = {"bounds", "summary", "params"}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _decode(row: Optional[sqlite3.Row]) -> Optional[dict]:
        if row is None:
            return None
        d = dict(row)
        for k in JSON_FIELDS & d.keys():
            if d[k]:
                d[k] = json.loads(d[k])
        return d

    def insert(self, table: str, **values: Any) -> int:
        values.setdefault("created", now())
        values = {k: json.dumps(v) if k in JSON_FIELDS and v is not None else v for k, v in values.items()}
        cols = ", ".join(values)
        qs = ", ".join("?" for _ in values)
        with self._lock, self._conn() as c:
            cur = c.execute(f"INSERT INTO {table} ({cols}) VALUES ({qs})", list(values.values()))
            return int(cur.lastrowid)

    def update(self, table: str, row_id: int, **values: Any) -> None:
        values = {k: json.dumps(v) if k in JSON_FIELDS and v is not None else v for k, v in values.items()}
        sets = ", ".join(f"{k} = ?" for k in values)
        with self._lock, self._conn() as c:
            c.execute(f"UPDATE {table} SET {sets} WHERE id = ?", [*values.values(), row_id])

    def get(self, table: str, row_id: int) -> Optional[dict]:
        with self._conn() as c:
            return self._decode(c.execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,)).fetchone())

    def all(self, table: str, where: str = "", args: tuple = (), limit: Optional[int] = None) -> list[dict]:
        sql = f"SELECT * FROM {table} {('WHERE ' + where) if where else ''} ORDER BY id DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._conn() as c:
            return [self._decode(r) for r in c.execute(sql, args).fetchall()]

    def delete(self, table: str, row_id: int) -> None:
        with self._lock, self._conn() as c:
            c.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
