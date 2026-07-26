"""Outbound HTTP restricted to an explicit host allowlist.

An agent with unrestricted HTTP is an SSRF primitive: it can reach cloud
metadata endpoints and internal services. This tool refuses anything not named
in STUDIO_HTTP_ALLOWED_HOSTS and blocks private address ranges outright.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from agentic_studio.agents.tools.registry import tool
from agentic_studio.settings import get_settings

MAX_BODY_CHARS = 20000
BLOCKED_HOSTNAMES = {"localhost", "metadata.google.internal", "metadata"}


def is_private_address(hostname: str) -> bool:
    try:
        resolved = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return True
    for entry in resolved:
        address = entry[4][0]
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            continue
        if parsed.is_private or parsed.is_loopback or parsed.is_link_local or parsed.is_reserved:
            return True
    return False


def check_url(url: str, allowed_hosts: list[str] | None = None) -> tuple[bool, str]:
    allowed = allowed_hosts if allowed_hosts is not None else get_settings().tools.http_allowed_hosts
    parsed = urllib.parse.urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        return False, f"scheme '{parsed.scheme}' is not allowed"
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False, "missing hostname"
    if hostname in BLOCKED_HOSTNAMES:
        return False, f"host '{hostname}' is blocked"
    if not allowed:
        return False, "no hosts are allowlisted; set STUDIO_HTTP_ALLOWED_HOSTS"
    if not any(hostname == entry.lower() or hostname.endswith(f".{entry.lower()}") for entry in allowed):
        return False, f"host '{hostname}' is not in the allowlist ({', '.join(allowed)})"
    if is_private_address(hostname):
        return False, f"host '{hostname}' resolves to a private or unroutable address"
    return True, ""


@tool(name="http_request", tags=("network",))
def http_request(url: str, method: str = "GET", body: str = "", headers: str = "") -> dict[str, Any]:
    """Call an allowlisted HTTP endpoint and return the status and body.

    Args:
        url: Absolute http(s) URL. The host must be allowlisted.
        method: HTTP method, GET or POST.
        body: Request body for POST, usually JSON.
        headers: Optional JSON object of extra request headers.
    """
    ok, reason = check_url(url)
    if not ok:
        return {"ok": False, "error": reason}

    method = method.upper()
    if method not in {"GET", "POST"}:
        return {"ok": False, "error": f"method '{method}' is not allowed"}

    extra_headers: dict[str, str] = {}
    if headers:
        try:
            parsed_headers = json.loads(headers)
            extra_headers = {str(k): str(v) for k, v in parsed_headers.items()}
        except Exception:
            return {"ok": False, "error": "headers must be a JSON object"}

    request = urllib.request.Request(
        url,
        data=body.encode("utf-8") if body and method == "POST" else None,
        headers={"User-Agent": "ai-agentic-studio/1.0", **extra_headers},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = response.read(MAX_BODY_CHARS * 2).decode("utf-8", errors="replace")
            return {
                "ok": True,
                "status": response.status,
                "content_type": response.headers.get("Content-Type", ""),
                "body": payload[:MAX_BODY_CHARS],
                "truncated": len(payload) > MAX_BODY_CHARS,
            }
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code,
                "error": exc.read().decode("utf-8", errors="replace")[:1000]}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
