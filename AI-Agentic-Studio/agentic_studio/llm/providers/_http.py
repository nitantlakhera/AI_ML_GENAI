"""Minimal JSON-over-HTTP helper.

Providers talk to vendor REST APIs directly through the standard library, so no
vendor SDK is required to run the studio. Streaming uses server-sent events.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from agentic_studio.core.errors import ProviderError


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    provider: str,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
        raise ProviderError(provider, f"HTTP {exc.code}: {body}", retryable=retryable) from exc
    except urllib.error.URLError as exc:
        raise ProviderError(provider, f"connection failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ProviderError(provider, "request timed out") from exc


def post_sse(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    provider: str,
) -> Iterator[dict[str, Any]]:
    """Yield decoded JSON events from a server-sent-events response."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    return
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise ProviderError(provider, f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(provider, f"connection failed: {exc.reason}") from exc


def post_jsonlines(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    provider: str,
) -> Iterator[dict[str, Any]]:
    """Yield decoded JSON objects from a newline-delimited stream (Ollama style)."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise ProviderError(provider, f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(provider, f"connection failed: {exc.reason}") from exc
