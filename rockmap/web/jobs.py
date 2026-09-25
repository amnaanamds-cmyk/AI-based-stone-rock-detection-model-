"""Persistent background job queue.

Jobs are rows in the ``jobs`` table. Worker threads (inside the web server, or in a
separate ``rockmap worker`` process) claim queued jobs atomically, report progress and a
heartbeat, write a log file and honour cancellation requests. Jobs interrupted by a crash or
restart are re-queued automatically (region jobs resume where they stopped, tile by tile).
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .db import Database, now

log = logging.getLogger("rockmap.jobs")
HANDLERS: dict[str, Callable[["JobContext"], None]] = {}
STALE_SECONDS = 180
MAX_ATTEMPTS = 3


class JobCancelled(Exception):
    pass


def handler(kind: str):
    def deco(fn):
        HANDLERS[kind] = fn
        return fn
    return deco


@dataclass
class JobContext:
    db: Database
    root: Path
    job: dict
    log_path: Path
    _last_beat: float = 0.0

    @property
    def params(self) -> dict:
        return self.job["params"] or {}

    def log(self, msg: str) -> None:
        with open(self.log_path, "a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    def progress(self, p: float, msg: str) -> None:
        t = time.time()
        row = self.db.get("jobs", self.job["id"])
        if row and row["cancel"]:
            raise JobCancelled()
        self.db.update("jobs", self.job["id"], progress=float(max(0.0, min(1.0, p))), message=msg, heartbeat=t)
        if t - self._last_beat > 5 or p >= 1:
            self.log(f"[{p * 100:5.1f}%] {msg}")
            self._last_beat = t


def enqueue(db: Database, kind: str, params: Optional[dict] = None, **cols) -> int:
    if kind not in HANDLERS:
        raise ValueError(f"unknown job kind {kind}")
    return db.insert("jobs", kind=kind, params=params or {}, status="queued", message="Queued", **cols)


def _claim(db: Database, worker_id: str) -> Optional[dict]:
    n = db.execute(
        "UPDATE jobs SET status='running', worker=?, started=?, heartbeat=?, attempts=attempts+1 "
        "WHERE id = (SELECT id FROM jobs WHERE status='queued' AND (region_id IS NULL OR region_id NOT IN "
        "(SELECT region_id FROM jobs WHERE status='running' AND region_id IS NOT NULL)) ORDER BY id LIMIT 1) "
        "AND status='queued'", (worker_id, now(), time.time()))
    if not n:
        return None
    return db.one("jobs", "worker = ? AND status = 'running' ORDER BY id DESC", (worker_id,))


def run_job(db: Database, root: Path, job: dict) -> None:
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    ctx = JobContext(db, root, job, logs / f"job{job['id']}.log")
    ctx.log(f"started {job['kind']} (attempt {job['attempts']}) params={job['params']}")
    done = threading.Event()

    def beat():  # keeps the job alive during long steps that report no progress
        while not done.wait(30):
            db.update("jobs", job["id"], heartbeat=time.time())
    threading.Thread(target=beat, daemon=True).start()
    try:
        HANDLERS[job["kind"]](ctx)
        db.update("jobs", job["id"], status="done", progress=1.0, finished=now(), message="Done")
        ctx.log("finished")
    except JobCancelled:
        db.update("jobs", job["id"], status="cancelled", finished=now(), message="Cancelled by user")
        ctx.log("cancelled")
    except Exception as e:  # noqa: BLE001 - report any failure to the user
        tb = traceback.format_exc()
        log.error("job %s failed: %s", job["id"], tb)
        ctx.log(tb)
        db.update("jobs", job["id"], status="failed", error=f"{type(e).__name__}: {e}", finished=now())
    finally:
        done.set()


def requeue_stale(db: Database) -> None:
    """Jobs whose worker died (no heartbeat) go back to the queue, up to MAX_ATTEMPTS."""
    cutoff = time.time() - STALE_SECONDS
    for j in db.all("jobs", "status = 'running' AND (heartbeat IS NULL OR heartbeat < ?)", (cutoff,)):
        if j["attempts"] < MAX_ATTEMPTS:
            db.update("jobs", j["id"], status="queued", worker=None, message="Re-queued after interruption")
        else:
            db.update("jobs", j["id"], status="failed", error="Interrupted too many times", finished=now())


class WorkerPool:
    def __init__(self, db: Database, root: Path, threads: int = 1, poll: float = 1.0):
        self.db, self.root, self.poll = db, root, poll
        self.threads = threads
        self._stop = threading.Event()
        self._workers: list[threading.Thread] = []

    def start(self) -> None:
        requeue_stale(self.db)
        for i in range(self.threads):
            t = threading.Thread(target=self._loop, name=f"rockmap-worker-{i}", daemon=True)
            t.start()
            self._workers.append(t)

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        wid = f"{uuid.uuid4().hex[:8]}"
        last_check = 0.0
        while not self._stop.is_set():
            try:
                if time.time() - last_check > 60:
                    requeue_stale(self.db)
                    last_check = time.time()
                job = _claim(self.db, wid)
            except Exception:  # noqa: BLE001 - database busy etc.
                log.exception("worker poll failed")
                job = None
            if job:
                run_job(self.db, self.root, job)
            else:
                self._stop.wait(self.poll)


def run_worker_forever(app, threads: int = 1) -> None:
    db = app.extensions["rockmap_db"]
    pool = WorkerPool(db, app.config["DATA_DIR"], threads)
    pool.start()
    print(f"RockMap worker running with {threads} thread(s); Ctrl+C to stop", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pool.stop()
