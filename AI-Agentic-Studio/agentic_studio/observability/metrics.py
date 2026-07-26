"""In-process counters, histograms, and token cost accounting.

Exposed over HTTP at `GET /metrics` so agent latency, tool failures, cache hit
rate, and spend are observable without an external backend.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

# USD per 1K tokens (input, output). Unknown models fall back to 0.0 so cost
# tracking never fabricates numbers.
COST_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4.1": (0.002, 0.008),
    "gpt-4.1-mini": (0.0004, 0.0016),
    "o3-mini": (0.0011, 0.0044),
    "claude-sonnet-4-20250514": (0.003, 0.015),
    "claude-3-5-haiku-20241022": (0.0008, 0.004),
    "gemini-2.0-flash": (0.0001, 0.0004),
    "gemini-1.5-pro": (0.00125, 0.005),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    key = model.strip()
    rates = COST_TABLE.get(key)
    if rates is None:
        for name, value in COST_TABLE.items():
            if key.startswith(name):
                rates = value
                break
    if rates is None:
        return 0.0
    return (prompt_tokens / 1000.0) * rates[0] + (completion_tokens / 1000.0) * rates[1]


def estimate_tokens(text: str) -> int:
    """Cheap token estimate; uses tiktoken when installed."""
    if not text:
        return 0
    try:  # pragma: no cover - optional dependency
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


@dataclass
class Histogram:
    values: list[float] = field(default_factory=list)

    def observe(self, value: float) -> None:
        self.values.append(value)

    def summary(self) -> dict[str, float]:
        if not self.values:
            return {"count": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        ordered = sorted(self.values)
        count = len(ordered)

        def pick(q: float) -> float:
            idx = min(count - 1, int(q * count))
            return ordered[idx]

        return {
            "count": count,
            "avg": round(sum(ordered) / count, 3),
            "p50": round(pick(0.5), 3),
            "p95": round(pick(0.95), 3),
            "max": round(ordered[-1], 3),
        }


class MetricsRegistry:
    """Thread-safe counters and histograms."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._histograms: dict[str, Histogram] = {}

    def incr(self, name: str, value: float = 1.0, **labels: Any) -> None:
        key = _key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def observe(self, name: str, value: float, **labels: Any) -> None:
        key = _key(name, labels)
        with self._lock:
            self._histograms.setdefault(key, Histogram()).observe(value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counters": {k: round(v, 6) for k, v in sorted(self._counters.items())},
                "histograms": {k: h.summary() for k, h in sorted(self._histograms.items())},
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histograms.clear()

    def counter(self, name: str, **labels: Any) -> float:
        return self._counters.get(_key(name, labels), 0.0)


def _key(name: str, labels: dict[str, Any]) -> str:
    if not labels:
        return name
    rendered = ",".join(f"{k}={v}" for k, v in sorted(labels.items()) if v is not None)
    return f"{name}{{{rendered}}}" if rendered else name


METRICS = MetricsRegistry()
