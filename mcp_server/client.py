"""MCP client for connecting to external MCP servers."""

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def list_tools(server_command: list[str]):
    """Connect to an MCP server and list available tools."""
    params = StdioServerParameters(command=server_command[0], args=server_command[1:])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return tools.tools
