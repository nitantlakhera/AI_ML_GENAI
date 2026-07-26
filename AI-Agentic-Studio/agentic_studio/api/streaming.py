"""Server-sent events helpers.

Streaming is the difference between a 12-second blank screen and visible
progress. Sources arrive first, then tokens, then a terminal `done` event.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

from fastapi.responses import StreamingResponse

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Disable proxy buffering so tokens are not held back until the response ends.
    "X-Accel-Buffering": "no",
}


def encode_event(payload: dict[str, Any], event: str | None = None) -> str:
    lines = []
    if event:
        lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(payload, default=str)}")
    return "\n".join(lines) + "\n\n"


def sse_stream(events: Iterable[dict[str, Any]]) -> Iterator[str]:
    """Wrap an event iterable, converting an exception into a terminal error event."""
    try:
        for payload in events:
            yield encode_event(payload, event=payload.get("type"))
    except Exception as exc:
        yield encode_event({"type": "error", "detail": f"{type(exc).__name__}: {exc}"}, event="error")
    yield encode_event({"type": "eof"}, event="eof")


def sse_response(events: Iterable[dict[str, Any]]) -> StreamingResponse:
    return StreamingResponse(
        sse_stream(events),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
