"""MCP client and the agent bridge.

`load_mcp_tools()` turns a remote MCP server into a list of `ToolSpec`s that the
agent runtime treats identically to local tools - same guardrails, same
approval gate, same tracing, same retries.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentic_studio.core.errors import StudioError, ToolError
from agentic_studio.core.types import ToolSpec
from agentic_studio.observability.logs import get_logger
from agentic_studio.observability.tracing import get_tracer

logger = get_logger("mcp.client")

DEFAULT_TIMEOUT_S = 30.0


class MCPUnavailable(StudioError):
    """The `mcp` package is not installed."""


@dataclass
class MCPServerConfig:
    """How to launch or reach one MCP server."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    prefix: str | None = None

    @property
    def tool_prefix(self) -> str:
        return self.prefix if self.prefix is not None else f"{self.name}_"

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> MCPServerConfig:
        return cls(
            name=name,
            command=data["command"],
            args=list(data.get("args", [])),
            env=dict(data.get("env", {})),
            prefix=data.get("prefix"),
        )

    def resolved_command(self) -> str:
        """Resolve the executable so a PATH miss fails loudly instead of silently."""
        found = shutil.which(self.command)
        if found is None:
            raise MCPUnavailable(f"MCP server command not found on PATH: {self.command}")
        return found


def load_config(path: Path) -> list[MCPServerConfig]:
    """Read a Claude/Cursor-style `mcpServers` config file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    servers = payload.get("mcpServers", payload)
    return [MCPServerConfig.from_dict(name, data) for name, data in servers.items()]


def _require_mcp() -> tuple[Any, Any, Any]:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:  # pragma: no cover - optional extra
        raise MCPUnavailable("the `mcp` package is not installed. Run: uv sync --extra mcp") from exc
    return ClientSession, StdioServerParameters, stdio_client


class MCPClient:
    """Talks to one MCP server over stdio.

    A session is opened per operation. That costs a process spawn per call but
    keeps the client stateless and safe to use from threads, which matters
    because the agent runtime executes tools in a thread pool.
    """

    def __init__(self, config: MCPServerConfig, timeout_s: float = DEFAULT_TIMEOUT_S):
        self.config = config
        self.timeout_s = timeout_s

    async def _session(self, operation: Any) -> Any:
        ClientSession, StdioServerParameters, stdio_client = _require_mcp()
        params = StdioServerParameters(
            command=self.config.resolved_command(),
            args=self.config.args,
            env=self.config.env or None,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await operation(session)

    async def alist_tools(self) -> list[dict[str, Any]]:
        async def operation(session: Any) -> list[dict[str, Any]]:
            listing = await session.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": getattr(tool, "inputSchema", None) or {"type": "object",
                                                                           "properties": {}},
                }
                for tool in listing.tools
            ]

        return await self._session(operation)

    async def acall_tool(self, name: str, arguments: dict[str, Any]) -> str:
        async def operation(session: Any) -> str:
            result = await session.call_tool(name, arguments)
            return _render_content(result)

        return await self._session(operation)

    def list_tools(self) -> list[dict[str, Any]]:
        return _run(self.alist_tools(), self.timeout_s)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return _run(self.acall_tool(name, arguments), self.timeout_s)


def _run(coroutine: Any, timeout_s: float) -> Any:
    """Run a coroutine from sync code, whether or not a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(asyncio.wait_for(coroutine, timeout=timeout_s))

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, asyncio.wait_for(coroutine, timeout=timeout_s))
        return future.result(timeout=timeout_s + 5)


def _render_content(result: Any) -> str:
    """Flatten an MCP tool result into text the model can read."""
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content or []:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
            continue
        data = getattr(item, "data", None)
        if data:
            parts.append(f"[{getattr(item, 'mimeType', 'binary')}: {len(str(data))} bytes]")
            continue
        parts.append(str(item))
    return "\n".join(parts) if parts else ""


def spec_from_mcp_tool(
    descriptor: dict[str, Any],
    caller: Any,
    prefix: str = "",
    requires_approval: bool = False,
    server_name: str = "mcp",
) -> ToolSpec:
    """Convert one MCP tool descriptor into a studio ToolSpec.

    `caller` is any callable taking (name, arguments) and returning text, which
    keeps this function pure and unit-testable without a live server.
    """
    remote_name = descriptor["name"]
    schema = descriptor.get("input_schema") or {"type": "object", "properties": {}}
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})

    def invoke(**arguments: Any) -> str:
        with get_tracer().span("tool.mcp", kind="tool", server=server_name, tool=remote_name):
            try:
                return caller(remote_name, arguments)
            except Exception as exc:
                raise ToolError(remote_name, f"MCP call failed: {exc}") from exc

    invoke.__name__ = f"{prefix}{remote_name}"
    return ToolSpec(
        name=f"{prefix}{remote_name}",
        description=(descriptor.get("description") or remote_name).strip(),
        parameters=schema,
        func=invoke,
        requires_approval=requires_approval,
        tags=("mcp", server_name),
    )


def load_mcp_tools(
    config: MCPServerConfig,
    requires_approval: bool = False,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> list[ToolSpec]:
    """Discover an MCP server's tools as agent-ready ToolSpecs."""
    client = MCPClient(config, timeout_s=timeout_s)
    descriptors = client.list_tools()
    specs = [
        spec_from_mcp_tool(
            descriptor,
            caller=client.call_tool,
            prefix=config.tool_prefix,
            requires_approval=requires_approval,
            server_name=config.name,
        )
        for descriptor in descriptors
    ]
    logger.info("loaded %d tool(s) from MCP server '%s'", len(specs), config.name)
    return specs


def register_mcp_tools(
    config: MCPServerConfig,
    registry: Any = None,
    requires_approval: bool = False,
) -> list[str]:
    """Load an MCP server's tools and register them for agents to use."""
    from agentic_studio.agents.tools.registry import REGISTRY

    registry = registry or REGISTRY
    names: list[str] = []
    for spec in load_mcp_tools(config, requires_approval=requires_approval):
        registry.register(spec)
        names.append(spec.name)
    return names


def register_from_config_file(path: Path, registry: Any = None) -> dict[str, list[str]]:
    """Register every server in an mcpServers config file, skipping failures."""
    registered: dict[str, list[str]] = {}
    for config in load_config(path):
        try:
            registered[config.name] = register_mcp_tools(config, registry=registry)
        except Exception as exc:
            logger.warning("skipping MCP server '%s': %s", config.name, exc)
            registered[config.name] = []
    return registered
