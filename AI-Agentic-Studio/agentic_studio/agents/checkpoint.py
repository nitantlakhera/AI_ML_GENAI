"""Graph checkpointers.

Checkpoints are what make an agent run survivable: a process restart, a crash,
or a pause for human approval can all be resumed from the last completed node
instead of replaying (and re-paying for) the whole run.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from agentic_studio.core.types import AgentStep, Message, ToolCall, ToolResult, Usage
from agentic_studio.settings import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id   TEXT PRIMARY KEY,
    next_node   TEXT NOT NULL,
    step        INTEGER NOT NULL,
    state       TEXT NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoint_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   TEXT NOT NULL,
    next_node   TEXT NOT NULL,
    step        INTEGER NOT NULL,
    state       TEXT NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_thread ON checkpoint_history(thread_id, step);
"""


class BaseCheckpointer(ABC):
    @abstractmethod
    def save(self, thread_id: str, state: dict[str, Any], next_node: str, step: int) -> None:
        ...

    @abstractmethod
    def load(self, thread_id: str) -> dict[str, Any] | None:
        ...

    @abstractmethod
    def delete(self, thread_id: str) -> bool:
        ...


class MemoryCheckpointer(BaseCheckpointer):
    """Process-local checkpoints. Fine for tests and single-run scripts."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def save(self, thread_id: str, state: dict[str, Any], next_node: str, step: int) -> None:
        with self._lock:
            self._data[thread_id] = {
                "state": dict(state),
                "next_node": next_node,
                "step": step,
                "updated_at": time.time(),
            }

    def load(self, thread_id: str) -> dict[str, Any] | None:
        with self._lock:
            snapshot = self._data.get(thread_id)
            return {**snapshot, "state": dict(snapshot["state"])} if snapshot else None

    def delete(self, thread_id: str) -> bool:
        with self._lock:
            return self._data.pop(thread_id, None) is not None


class SqliteCheckpointer(BaseCheckpointer):
    """Durable checkpoints with a full step history for replay and debugging."""

    def __init__(self, path: Path | None = None, keep_history: bool = True):
        self.path = Path(path or get_settings().paths.checkpoints_db)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.keep_history = keep_history
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, thread_id: str, state: dict[str, Any], next_node: str, step: int) -> None:
        payload = json.dumps(_serialize(state), default=str)
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO checkpoints (thread_id, next_node, step, state, updated_at)"
                " VALUES (?, ?, ?, ?, ?) ON CONFLICT(thread_id) DO UPDATE SET"
                " next_node = ?, step = ?, state = ?, updated_at = ?",
                (thread_id, next_node, step, payload, now, next_node, step, payload, now),
            )
            if self.keep_history:
                conn.execute(
                    "INSERT INTO checkpoint_history (thread_id, next_node, step, state, updated_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (thread_id, next_node, step, payload, now),
                )

    def load(self, thread_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT next_node, step, state, updated_at FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "state": _deserialize(json.loads(row["state"])),
            "next_node": row["next_node"],
            "step": int(row["step"]),
            "updated_at": row["updated_at"],
        }

    def history(self, thread_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT next_node, step, updated_at FROM checkpoint_history"
                " WHERE thread_id = ? ORDER BY step LIMIT ?",
                (thread_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete(self, thread_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM checkpoint_history WHERE thread_id = ?", (thread_id,))
            return cursor.rowcount > 0

    def list_threads(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT thread_id, next_node, step, updated_at FROM checkpoints"
                " ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]


# Graph state carries dataclasses (messages, steps, usage). Tagging them on the
# way out and rebuilding on the way in means a resumed run keeps real objects
# instead of stringified debris.
_TAG = "__type__"


def _encode(value: Any) -> Any:
    if isinstance(value, Message):
        return {_TAG: "Message", "v": value.to_dict()}
    if isinstance(value, ToolCall):
        return {_TAG: "ToolCall", "v": value.to_dict()}
    if isinstance(value, ToolResult):
        return {_TAG: "ToolResult", "v": value.to_dict()}
    if isinstance(value, Usage):
        return {_TAG: "Usage", "v": value.to_dict()}
    if isinstance(value, AgentStep):
        return {
            _TAG: "AgentStep",
            "v": {
                "index": value.index,
                "node": value.node,
                "thought": value.thought,
                "tool_calls": [_encode(call) for call in value.tool_calls],
                "results": [_encode(result) for result in value.results],
            },
        }
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value

    tag = value.get(_TAG)
    if tag is None:
        return {key: _decode(item) for key, item in value.items()}

    payload = value["v"]
    if tag == "Message":
        return Message.from_dict(payload)
    if tag == "ToolCall":
        return ToolCall(name=payload["name"], arguments=payload.get("arguments", {}),
                        id=payload.get("id", ""))
    if tag == "ToolResult":
        return ToolResult(
            tool_call_id=payload.get("tool_call_id", ""),
            name=payload.get("name", ""),
            output=payload.get("output", ""),
            ok=payload.get("ok", True),
            error=payload.get("error"),
            latency_ms=payload.get("latency_ms", 0.0),
        )
    if tag == "Usage":
        return Usage(
            prompt_tokens=payload.get("prompt_tokens", 0),
            completion_tokens=payload.get("completion_tokens", 0),
            cost_usd=payload.get("cost_usd", 0.0),
        )
    if tag == "AgentStep":
        return AgentStep(
            index=payload.get("index", 0),
            node=payload.get("node", ""),
            thought=payload.get("thought", ""),
            tool_calls=[_decode(call) for call in payload.get("tool_calls", [])],
            results=[_decode(result) for result in payload.get("results", [])],
        )
    return payload


def _serialize(state: dict[str, Any]) -> dict[str, Any]:
    return {key: _encode(value) for key, value in state.items()}


def _deserialize(state: dict[str, Any]) -> dict[str, Any]:
    return {key: _decode(value) for key, value in state.items()}
