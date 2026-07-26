"""Background jobs.

Ingestion and evaluation take minutes, which is longer than any sensible HTTP
timeout. Those endpoints enqueue a job, return immediately with a job id, and
the caller polls `/jobs/{id}`. State lives in SQLite so a restart does not lose
the record of what ran.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agentic_studio.core.types import new_id
from agentic_studio.observability.logs import get_logger
from agentic_studio.observability.metrics import METRICS
from agentic_studio.settings import get_settings

logger = get_logger("api.jobs")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  REAL NOT NULL,
    finished_at REAL,
    result      TEXT,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at);
"""

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"


class JobStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or get_settings().paths.jobs_db)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def create(self, kind: str) -> str:
        job_id = new_id("job")
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, kind, status, created_at) VALUES (?, ?, ?, ?)",
                (job_id, kind, QUEUED, time.time()),
            )
        METRICS.incr("jobs_created", kind=kind)
        return job_id

    def mark(
        self,
        job_id: str,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        finished = time.time() if status in {SUCCEEDED, FAILED} else None
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, result = ?, error = ?, finished_at = ? WHERE job_id = ?",
                (
                    status,
                    json.dumps(result, default=str) if result is not None else None,
                    error,
                    finished,
                    job_id,
                ),
            )
        if status in {SUCCEEDED, FAILED}:
            METRICS.incr("jobs_finished", status=status)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return _to_dict(row) if row else None

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_to_dict(row) for row in rows]

    def clear(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM jobs")

    def run(self, job_id: str, work: Callable[[], dict[str, Any]]) -> None:
        """Execute a job, recording success or the failure reason."""
        self.mark(job_id, RUNNING)
        try:
            result = work()
            self.mark(job_id, SUCCEEDED, result=result)
        except Exception as exc:
            logger.warning("job %s failed: %s", job_id, exc)
            self.mark(
                job_id,
                FAILED,
                error=f"{type(exc).__name__}: {exc}",
                result={"traceback": traceback.format_exc()[-2000:]},
            )


def _to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    if data.get("result"):
        try:
            data["result"] = json.loads(data["result"])
        except Exception:
            pass
    return data


_STORE: JobStore | None = None


def get_job_store() -> JobStore:
    global _STORE
    if _STORE is None:
        _STORE = JobStore()
    return _STORE


def reset_job_store() -> None:
    global _STORE
    _STORE = None
