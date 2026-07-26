"""MCP server publishing the studio's tools.

Rather than hand-writing MCP wrappers, this walks the studio tool registry and
publishes each safe tool, so a tool added for the agent is automatically
available to Cursor and other MCP clients with no extra code.

Run it with:  uv run python -m agentic_studio.mcp_bridge.server
"""

from __future__ import annotations

import json
from typing import Any

from agentic_studio.agents.tools import REGISTRY, SAFE_TOOL_NAMES
from agentic_studio.core.types import ToolCall
from agentic_studio.observability.logs import get_logger

logger = get_logger("mcp.server")

SERVER_NAME = "ai-agentic-studio"

# Tools that mutate state or execute code are not published over MCP; an MCP
# client cannot pass through the studio's human-approval gate.
PUBLISHED_TOOLS = SAFE_TOOL_NAMES


def build_server(tool_names: tuple[str, ...] = PUBLISHED_TOOLS) -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError("the `mcp` package is not installed. Run: uv sync --extra mcp") from exc

    server = FastMCP(SERVER_NAME)

    for spec in REGISTRY.specs(allow=tool_names):
        _publish(server, spec)

    @server.tool()
    def studio_info() -> str:
        """Describe this studio instance: providers, index size, and available tools."""
        from agentic_studio.llm.router import get_router
        from agentic_studio.rag.pipeline import get_pipeline

        return json.dumps(
            {
                "server": SERVER_NAME,
                "providers": get_router().describe(),
                "corpus": get_pipeline().stats(),
                "tools": [s.name for s in REGISTRY.specs(allow=tool_names)],
            },
            indent=2,
            default=str,
        )

    logger.info("MCP server exposing %d tool(s)", len(tool_names) + 1)
    return server


def _publish(server: Any, spec: Any) -> None:
    """Register one studio tool with FastMCP, routing through the registry.

    Going through `REGISTRY.run` rather than calling the function directly means
    MCP callers get the same timeouts, retries, and trace spans as agents.
    """

    def handler(arguments: dict[str, Any] | None = None) -> str:
        result = REGISTRY.run(ToolCall(name=spec.name, arguments=arguments or {}))
        return result.output if result.ok else f"ERROR: {result.error}"

    handler.__name__ = spec.name
    handler.__doc__ = (
        f"{spec.description}\n\nArguments (JSON object): "
        f"{json.dumps(spec.parameters.get('properties', {}))}"
    )
    server.tool(name=spec.name)(handler)


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
