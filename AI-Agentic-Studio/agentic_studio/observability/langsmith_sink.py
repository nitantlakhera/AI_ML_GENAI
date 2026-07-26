"""Optional LangSmith span forwarding over plain HTTP (no SDK required)."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from agentic_studio.settings import get_settings

_ENDPOINT = "https://api.smith.langchain.com/runs"


def send(payload: dict[str, Any], timeout: float = 3.0) -> bool:
    settings = get_settings()
    key = settings.observability.langsmith_api_key
    if not key:
        return False

    body = {
        "name": payload["name"],
        "run_type": _run_type(payload.get("kind", "internal")),
        "inputs": {"attributes": payload.get("attributes", {})},
        "outputs": {"status": payload.get("status")},
        "extra": {"trace_id": payload.get("trace_id"), "duration_ms": payload.get("duration_ms")},
        "session_name": settings.observability.langsmith_project,
        "error": payload.get("error"),
    }
    request = urllib.request.Request(
        _ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-api-key": key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return True
    except Exception:
        return False


def _run_type(kind: str) -> str:
    return {
        "llm": "llm",
        "tool": "tool",
        "retriever": "retriever",
        "chain": "chain",
        "agent": "chain",
    }.get(kind, "chain")
