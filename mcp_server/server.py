"""MCP server exposing project tools via Model Context Protocol."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ai-ml-genai")


@mcp.tool()
def greet(name: str) -> str:
    """Return a greeting for the given name."""
    return f"Hello, {name}!"


@mcp.tool()
def word_count(text: str) -> int:
    """Count words in the given text."""
    return len(text.split())


if __name__ == "__main__":
    mcp.run()
