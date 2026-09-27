"""SQLite persistence: users, scenes, regions, models, jobs, annotations and the audit log.

WAL mode lets the web server and a separate ``rockmap worker`` process share the database.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Iterator, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'analyst',
    active        INTEGER NOT NULL DEFAULT 1,
    api_token_hash TEXT,
    must_change   INTEGER NOT NULL DEFAULT 0,
    last_login    TEXT,
    created       TEXT NOT NULL
);
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
    created_by    TEXT,
    created       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS regions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    folder      TEXT NOT NULL,
    preset      TEXT,
    bounds      TEXT,
    share_token TEXT,
    created_by  TEXT,
    created     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT NOT NULL,
    folder    TEXT NOT NULL,
    scene_id  INTEGER,
    region_id INTEGER,
    uses_dem  INTEGER NOT NULL DEFAULT 0,
    best_algo TEXT,
    summary   TEXT,
    created_by TEXT,
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
    region_id INTEGER,
    model_id  INTEGER,
    folder    TEXT,
    error     TEXT,
    user      TEXT,
    worker    TEXT,
    cancel    INTEGER NOT NULL DEFAULT 0,
    attempts  INTEGER NOT NULL DEFAULT 0,
    heartbeat REAL,
    started   TEXT,
    created   TEXT NOT NULL,
    finished  TEXT
);
CREATE TABLE IF NOT EXISTS annotations (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    region_id INTEGER NOT NULL,
    class_id  INTEGER NOT NULL,
    geometry  TEXT NOT NULL,
    note      TEXT,
    author    TEXT,
    created   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    region_id   INTEGER,
    lat         REAL NOT NULL,
    lon         REAL NOT NULL,
    accuracy_m  REAL,
    class_id    INTEGER NOT NULL,
    certainty   INTEGER NOT NULL DEFAULT 2,
    note        TEXT,
    photo       TEXT,
    gem         TEXT,
    observed_at TEXT,
    author      TEXT,
    created     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    action   TEXT NOT NULL,
    detail   TEXT,
    ip       TEXT,
    created  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS ann_region ON annotations(region_id);
CREATE INDEX IF NOT EXISTS obs_region ON observations(region_id);
"""

# columns added after v1.0 (databases created by older versions are migrated in place)
MIGRATIONS = {
    "scenes": {"created_by": "TEXT"},
    "regions": {"share_token": "TEXT"},
    "users": {"must_change": "INTEGER NOT NULL DEFAULT 0"},
    "observations": {"gem": "TEXT"},
    "models": {"region_id": "INTEGER", "created_by": "TEXT"},
    "jobs": {"region_id": "INTEGER", "user": "TEXT", "worker": "TEXT", "cancel": "INTEGER NOT NULL DEFAULT 0",
             "attempts": "INTEGER NOT NULL DEFAULT 0", "heartbeat": "REAL", "started": "TEXT"},
}

JSON_FIELDS = {"bounds", "summary", "params", "geometry"}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            # old databases: add new columns before creating indexes that use them
            existing = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table, cols in MIGRATIONS.items():
                if table in existing:
                    have = {r[1] for r in c.execute(f"PRAGMA table_info({table})")}
                    for col, decl in cols.items():
                        if col not in have:
                            c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """Connection that commits (or rolls back) and is always closed - Windows keeps open files locked."""
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _decode(row: Optional[sqlite3.Row]) -> Optional[dict]:
        if row is None:
            return None
        d = dict(row)
        for k in JSON_FIELDS & d.keys():
            if d[k]:
                d[k] = json.loads(d[k])
        return d

    @staticmethod
    def _encode(values: dict) -> dict:
        return {k: json.dumps(v) if k in JSON_FIELDS and v is not None else v for k, v in values.items()}

    def insert(self, table: str, **values: Any) -> int:
        values.setdefault("created", now())
        values = self._encode(values)
        cols = ", ".join(values)
        qs = ", ".join("?" for _ in values)
        with self._lock, self._conn() as c:
            cur = c.execute(f"INSERT INTO {table} ({cols}) VALUES ({qs})", list(values.values()))
            return int(cur.lastrowid)

    def update(self, table: str, row_id: int, **values: Any) -> None:
        values = self._encode(values)
        sets = ", ".join(f"{k} = ?" for k in values)
        with self._lock, self._conn() as c:
            c.execute(f"UPDATE {table} SET {sets} WHERE id = ?", [*values.values(), row_id])

    def execute(self, sql: str, args: tuple = ()) -> int:
        with self._lock, self._conn() as c:
            return c.execute(sql, args).rowcount

    def get(self, table: str, row_id: int) -> Optional[dict]:
        with self._conn() as c:
            return self._decode(c.execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,)).fetchone())

    def one(self, table: str, where: str, args: tuple = ()) -> Optional[dict]:
        with self._conn() as c:
            return self._decode(c.execute(f"SELECT * FROM {table} WHERE {where} LIMIT 1", args).fetchone())

    def all(self, table: str, where: str = "", args: tuple = (), limit: Optional[int] = None,
            order: str = "id DESC") -> list[dict]:
        sql = f"SELECT * FROM {table} {('WHERE ' + where) if where else ''} ORDER BY {order}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._conn() as c:
            return [self._decode(r) for r in c.execute(sql, args).fetchall()]

    def count(self, table: str, where: str = "", args: tuple = ()) -> int:
        with self._conn() as c:
            return int(c.execute(f"SELECT COUNT(*) FROM {table} {('WHERE ' + where) if where else ''}",
                                 args).fetchone()[0])

    def delete(self, table: str, row_id: int) -> None:
        with self._lock, self._conn() as c:
            c.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
