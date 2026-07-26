"""Human-in-the-loop approvals.

An agent that can write files, execute code, or call a paid API needs a gate.
When a tool is marked `requires_approval`, the run pauses, an approval request is
persisted, and the graph is checkpointed. A human approves or rejects through the
API or UI, and the run resumes exactly where it stopped.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from agentic_studio.core.types import new_id
from agentic_studio.observability.metrics import METRICS
from agentic_studio.settings import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS approvals (
    request_id  TEXT PRIMARY KEY,
    thread_id   TEXT NOT NULL,
    tool        TEXT NOT NULL,
    arguments   TEXT NOT NULL,
    status      TEXT NOT NULL,
    reason      TEXT,
    decided_by  TEXT,
    created_at  REAL NOT NULL,
    decided_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_approvals_thread ON approvals(thread_id, status);
"""

PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"


class ApprovalStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or get_settings().paths.approvals_db)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def create(self, thread_id: str, tool: str, arguments: dict[str, Any]) -> str:
        request_id = new_id("appr")
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO approvals (request_id, thread_id, tool, arguments, status, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (request_id, thread_id, tool, json.dumps(arguments, default=str), PENDING, time.time()),
            )
        METRICS.incr("approvals_requested", tool=tool)
        return request_id

    def get(self, request_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM approvals WHERE request_id = ?", (request_id,)
            ).fetchone()
        return _to_dict(row) if row else None

    def decide(
        self, request_id: str, approved: bool, reason: str = "", decided_by: str = "human"
    ) -> dict[str, Any] | None:
        status = APPROVED if approved else REJECTED
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "UPDATE approvals SET status = ?, reason = ?, decided_by = ?, decided_at = ?"
                " WHERE request_id = ? AND status = ?",
                (status, reason, decided_by, time.time(), request_id, PENDING),
            )
            if cursor.rowcount == 0:
                return None
        METRICS.incr("approvals_decided", status=status)
        return self.get(request_id)

    def pending(self, thread_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        query = "SELECT * FROM approvals WHERE status = ?"
        params: list[Any] = [PENDING]
        if thread_id:
            query += " AND thread_id = ?"
            params.append(thread_id)
        query += " ORDER BY created_at LIMIT ?"
        params.append(limit)
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_to_dict(row) for row in rows]

    def history(self, thread_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM approvals WHERE thread_id = ? ORDER BY created_at DESC LIMIT ?",
                (thread_id, limit),
            ).fetchall()
        return [_to_dict(row) for row in rows]

    def clear(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM approvals")


def _to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    try:
        data["arguments"] = json.loads(data["arguments"])
    except Exception:
        pass
    return data


_STORE: ApprovalStore | None = None


def get_approval_store() -> ApprovalStore:
    global _STORE
    if _STORE is None:
        _STORE = ApprovalStore()
    return _STORE


def set_approval_store(store: ApprovalStore) -> None:
    global _STORE
    _STORE = store


def reset_approval_store() -> None:
    global _STORE
    _STORE = None
