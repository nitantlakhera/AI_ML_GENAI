"""Hierarchical tracing for LLM calls, retrieval stages, tools, and agent nodes.

Traces are always available in memory (`tracer.recent()`), optionally appended
to JSONL, and optionally forwarded to LangSmith or an OTLP collector when those
environment variables are set. No external service is required.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentic_studio.settings import get_settings

_local = threading.local()


@dataclass
class Span:
    name: str
    kind: str = "internal"
    trace_id: str = ""
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: str | None = None
    start_ms: float = field(default_factory=lambda: time.time() * 1000)
    end_ms: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        end = self.end_ms if self.end_ms is not None else time.time() * 1000
        return end - self.start_ms

    def set(self, **attributes: Any) -> Span:
        self.attributes.update(attributes)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
        }


class Tracer:
    """Collects spans; sinks are additive and all optional."""

    def __init__(self, sink: str | None = None, jsonl_path: Path | None = None, enabled: bool | None = None):
        settings = get_settings()
        self.enabled = settings.observability.tracing_enabled if enabled is None else enabled
        self.sink = sink or settings.observability.trace_sink
        self.jsonl_path = jsonl_path or settings.paths.traces
        self._buffer: deque[dict[str, Any]] = deque(maxlen=2000)
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def span(self, name: str, kind: str = "internal", **attributes: Any) -> Iterator[Span]:
        parent = getattr(_local, "current_span", None)
        current = Span(
            name=name,
            kind=kind,
            trace_id=parent.trace_id if parent else uuid.uuid4().hex[:16],
            parent_id=parent.span_id if parent else None,
            attributes=dict(attributes),
        )
        if not self.enabled:
            yield current
            return

        _local.current_span = current
        try:
            yield current
        except Exception as exc:
            current.status = "error"
            current.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            current.end_ms = time.time() * 1000
            _local.current_span = parent
            self._emit(current)

    def _emit(self, span_obj: Span) -> None:
        payload = span_obj.to_dict()
        with self._lock:
            self._buffer.append(payload)
        if self.sink == "jsonl":
            try:
                self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                with self.jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, default=str) + "\n")
            except OSError:
                pass
        _forward_external(payload)

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._buffer)[-limit:]

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def trace_tree(self, trace_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [s for s in self._buffer if s["trace_id"] == trace_id]


def _forward_external(payload: dict[str, Any]) -> None:
    """Best-effort forwarding to LangSmith / OTLP when configured."""
    settings = get_settings()
    if settings.observability.langsmith_api_key:
        try:  # pragma: no cover - network path
            from agentic_studio.observability.langsmith_sink import send

            send(payload)
        except Exception:
            pass
    if settings.observability.otel_endpoint:
        try:  # pragma: no cover - network path
            from agentic_studio.observability.otel_sink import send

            send(payload)
        except Exception:
            pass


_TRACER: Tracer | None = None


def get_tracer() -> Tracer:
    global _TRACER
    if _TRACER is None:
        _TRACER = Tracer()
    return _TRACER


def set_tracer(tracer: Tracer) -> None:
    global _TRACER
    _TRACER = tracer


def span(name: str, kind: str = "internal", **attributes: Any):
    return get_tracer().span(name, kind, **attributes)


def current_trace_id() -> str | None:
    current = getattr(_local, "current_span", None)
    return current.trace_id if current else None
