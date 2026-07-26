"""Exact and semantic response cache backed by SQLite.

Exact hits key on the full request fingerprint. Semantic hits compare the
embedding of the final user turn against previous requests for the same model,
which collapses paraphrased questions onto one paid call.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from agentic_studio.core.types import LLMResponse, Message, ToolCall, Usage
from agentic_studio.settings import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    key         TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    probe       TEXT NOT NULL,
    embedding   TEXT,
    response    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_cache_model ON llm_cache(model, created_at);
"""


class ResponseCache:
    def __init__(
        self,
        path: Path | None = None,
        ttl_s: int | None = None,
        semantic: bool | None = None,
        similarity_threshold: float | None = None,
    ):
        settings = get_settings()
        self.path = path or settings.paths.cache_db
        self.ttl_s = settings.cache.ttl_s if ttl_s is None else ttl_s
        self.semantic = settings.cache.semantic if semantic is None else semantic
        self.threshold = (
            settings.cache.similarity_threshold if similarity_threshold is None else similarity_threshold
        )
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def fingerprint(messages: list[Message], model: str, **options: Any) -> str:
        payload = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "options": {k: v for k, v in sorted(options.items()) if v is not None},
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, messages: list[Message], model: str, **options: Any) -> LLMResponse | None:
        key = self.fingerprint(messages, model, **options)
        cutoff = time.time() - self.ttl_s

        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT response FROM llm_cache WHERE key = ? AND created_at > ?", (key, cutoff)
            ).fetchone()
            if row:
                return _deserialize(row["response"])

            if not self.semantic:
                return None

            probe = _probe_text(messages)
            vector = _embed(probe)
            if vector is None:
                return None

            rows = conn.execute(
                "SELECT embedding, response FROM llm_cache WHERE model = ? AND created_at > ?",
                (model, cutoff),
            ).fetchall()

        best_score = 0.0
        best_response: str | None = None
        for candidate in rows:
            stored = candidate["embedding"]
            if not stored:
                continue
            score = _cosine(vector, json.loads(stored))
            if score > best_score:
                best_score = score
                best_response = candidate["response"]

        if best_response is not None and best_score >= self.threshold:
            return _deserialize(best_response)
        return None

    def set(self, messages: list[Message], model: str, response: LLMResponse, **options: Any) -> None:
        key = self.fingerprint(messages, model, **options)
        probe = _probe_text(messages)
        vector = _embed(probe) if self.semantic else None
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, model, created_at, probe, embedding, response)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    key,
                    model,
                    time.time(),
                    probe[:2000],
                    json.dumps(vector) if vector is not None else None,
                    json.dumps(response.to_dict()),
                ),
            )

    def clear(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM llm_cache")

    def size(self) -> int:
        with self._lock, self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0])


def _probe_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user" and message.content:
            return message.content
    return messages[-1].content if messages else ""


def _embed(text: str) -> list[float] | None:
    """Imported lazily to keep the llm package independent of the rag package."""
    if not text.strip():
        return None
    try:
        from agentic_studio.rag.embeddings import get_embedder

        return [float(value) for value in get_embedder().embed_query(text)]
    except Exception:
        return None


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = sum(a * a for a in left) ** 0.5
    norm_right = sum(b * b for b in right) ** 0.5
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot / (norm_left * norm_right)


def _deserialize(payload: str) -> LLMResponse:
    data = json.loads(payload)
    usage = data.get("usage", {})
    return LLMResponse(
        text=data.get("text", ""),
        provider=data.get("provider", ""),
        model=data.get("model", ""),
        usage=Usage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            cost_usd=0.0,
        ),
        tool_calls=[
            ToolCall(name=c["name"], arguments=c.get("arguments", {})) for c in data.get("tool_calls", [])
        ],
        latency_ms=0.0,
        cached=True,
        finish_reason=data.get("finish_reason", "stop"),
    )
