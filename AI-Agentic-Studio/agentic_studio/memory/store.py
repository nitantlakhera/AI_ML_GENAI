"""Durable conversation memory on SQLite.

In-process dictionaries lose every conversation on restart and cannot be shared
between the API, the UI, and a background worker. A single file-backed store
fixes both, and swapping in Postgres or Redis means reimplementing this one
class.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from agentic_studio.core.types import Message, ToolCall
from agentic_studio.settings import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id   TEXT PRIMARY KEY,
    title       TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    metadata    TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    name        TEXT,
    tool_call_id TEXT,
    tool_calls  TEXT,
    images      TEXT,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, seq);
CREATE TABLE IF NOT EXISTS summaries (
    thread_id   TEXT PRIMARY KEY,
    summary     TEXT NOT NULL,
    upto_seq    INTEGER NOT NULL,
    updated_at  REAL NOT NULL
);
"""


class ConversationStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or get_settings().paths.memory_db)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # -- threads ------------------------------------------------------------

    def ensure_thread(self, thread_id: str, title: str | None = None,
                      metadata: dict[str, Any] | None = None) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO threads (thread_id, title, created_at, updated_at, metadata) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(thread_id) DO UPDATE SET updated_at = ?",
                (thread_id, title, now, now, json.dumps(metadata or {}), now),
            )

    def list_threads(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT t.thread_id, t.title, t.created_at, t.updated_at,"
                " (SELECT COUNT(*) FROM messages m WHERE m.thread_id = t.thread_id) AS messages"
                " FROM threads t ORDER BY t.updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_thread(self, thread_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM summaries WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
            return cursor.rowcount > 0

    # -- messages -----------------------------------------------------------

    def append(self, thread_id: str, message: Message) -> int:
        self.ensure_thread(thread_id)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM messages WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            seq = int(row[0]) + 1
            conn.execute(
                "INSERT INTO messages (thread_id, seq, role, content, name, tool_call_id,"
                " tool_calls, images, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    thread_id,
                    seq,
                    message.role,
                    message.content,
                    message.name,
                    message.tool_call_id,
                    json.dumps([c.to_dict() for c in message.tool_calls]) if message.tool_calls else None,
                    json.dumps(message.images) if message.images else None,
                    time.time(),
                ),
            )
            conn.execute("UPDATE threads SET updated_at = ? WHERE thread_id = ?", (time.time(), thread_id))
            return seq

    def extend(self, thread_id: str, messages: list[Message]) -> int:
        last = 0
        for message in messages:
            last = self.append(thread_id, message)
        return last

    def history(self, thread_id: str, limit: int | None = None, after_seq: int = 0) -> list[Message]:
        query = "SELECT * FROM messages WHERE thread_id = ? AND seq > ? ORDER BY seq"
        params: list[Any] = [thread_id, after_seq]
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        messages = [_to_message(row) for row in rows]
        return messages[-limit:] if limit else messages

    def message_count(self, thread_id: str) -> int:
        with self._lock, self._connect() as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE thread_id = ?", (thread_id,)
                ).fetchone()[0]
            )

    # -- summaries ----------------------------------------------------------

    def get_summary(self, thread_id: str) -> tuple[str, int] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT summary, upto_seq FROM summaries WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        return (row["summary"], int(row["upto_seq"])) if row else None

    def set_summary(self, thread_id: str, summary: str, upto_seq: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO summaries (thread_id, summary, upto_seq, updated_at) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(thread_id) DO UPDATE SET summary = ?, upto_seq = ?, updated_at = ?",
                (thread_id, summary, upto_seq, time.time(), summary, upto_seq, time.time()),
            )

    def clear(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript("DELETE FROM messages; DELETE FROM summaries; DELETE FROM threads;")


def _to_message(row: sqlite3.Row) -> Message:
    tool_calls = []
    if row["tool_calls"]:
        tool_calls = [
            ToolCall(name=c["name"], arguments=c.get("arguments", {}), id=c.get("id", ""))
            for c in json.loads(row["tool_calls"])
        ]
    return Message(
        role=row["role"],
        content=row["content"],
        name=row["name"],
        tool_call_id=row["tool_call_id"],
        tool_calls=tool_calls,
        images=json.loads(row["images"]) if row["images"] else [],
    )


_STORE: ConversationStore | None = None


def get_store() -> ConversationStore:
    global _STORE
    if _STORE is None:
        _STORE = ConversationStore()
    return _STORE


def set_store(store: ConversationStore) -> None:
    global _STORE
    _STORE = store


def reset_store() -> None:
    global _STORE
    _STORE = None
