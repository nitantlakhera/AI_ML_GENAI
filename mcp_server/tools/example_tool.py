"""Example MCP tool definitions (register in mcp_server/server.py)."""


def format_document_summary(title: str, content: str, max_length: int = 200) -> str:
    """Format a short document summary for MCP tool responses."""
    preview = content[:max_length]
    suffix = "..." if len(content) > max_length else ""
    return f"{title}: {preview}{suffix}"
