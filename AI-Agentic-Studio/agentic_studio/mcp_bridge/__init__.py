"""Model Context Protocol integration, in both directions.

Outward: `server.py` publishes the studio's own tools (RAG search, graph
explore, corpus stats) so Cursor or any MCP client can use them.

Inward: `client.py` connects to external MCP servers and converts their tools
into `ToolSpec`s, which means an agent can call them like any built-in tool.
That inward path is what was missing when MCP and agents lived side by side
without ever meeting.

The package is named `mcp_bridge`, not `mcp`, so it cannot shadow the PyPI
`mcp` package.
"""

from agentic_studio.mcp_bridge.client import (
    MCPClient,
    MCPServerConfig,
    load_mcp_tools,
    register_mcp_tools,
    spec_from_mcp_tool,
)

__all__ = [
    "MCPClient",
    "MCPServerConfig",
    "load_mcp_tools",
    "register_mcp_tools",
    "spec_from_mcp_tool",
]
