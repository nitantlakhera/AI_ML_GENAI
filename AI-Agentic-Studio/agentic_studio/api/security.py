"""API-key authentication and per-caller rate limiting.

Auth activates only when STUDIO_API_KEYS is set, so local development stays
frictionless while a deployed instance is closed by configuration rather than by
a code change.
"""

from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request, status

from agentic_studio.observability.metrics import METRICS
from agentic_studio.settings import get_settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """FastAPI dependency: validate the X-API-Key header when auth is enabled."""
    settings = get_settings().api
    if not settings.auth_enabled:
        return "anonymous"
    if not x_api_key:
        METRICS.incr("api_auth_missing")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    for candidate in settings.api_keys:
        if hmac.compare_digest(candidate, x_api_key):
            return x_api_key[:8]
    METRICS.incr("api_auth_rejected")
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid API key")


class RateLimiter:
    """Sliding-window limiter keyed by API key, falling back to client IP."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int, float]:
        """Return (allowed, remaining, retry_after_seconds)."""
        if self.per_minute <= 0:
            return True, 0, 0.0
        now = time.time()
        window_start = now - 60.0
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] < window_start:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return False, 0, max(0.0, 60.0 - (now - hits[0]))
            hits.append(now)
            return True, self.per_minute - len(hits), 0.0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


_LIMITER: RateLimiter | None = None


def get_limiter() -> RateLimiter:
    global _LIMITER
    if _LIMITER is None:
        _LIMITER = RateLimiter(get_settings().api.rate_limit_per_minute)
    return _LIMITER


def caller_key(request: Request) -> str:
    api_key = request.headers.get("x-api-key")
    if api_key:
        return f"key:{api_key[:12]}"
    client = request.client.host if request.client else "unknown"
    return f"ip:{client}"


EXEMPT_PATHS = {"/health", "/docs", "/redoc", "/openapi.json", "/metrics"}


async def rate_limit_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    if request.url.path in EXEMPT_PATHS:
        return await call_next(request)

    allowed, remaining, retry_after = get_limiter().check(caller_key(request))
    if not allowed:
        from fastapi.responses import JSONResponse

        METRICS.incr("api_rate_limited")
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "rate limit exceeded", "retry_after_s": round(retry_after, 1)},
            headers={"Retry-After": str(int(retry_after) + 1)},
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    return response
