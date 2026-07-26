"""Built-in tool suite.

Importing this package registers every tool. `default_tools()` returns the safe
set for a general-purpose agent; dangerous tools stay available but must be
opted into (and pass the approval gate).
"""

from __future__ import annotations

from agentic_studio.core.types import ToolSpec

# Importing these modules is what registers the tools.
from agentic_studio.agents.tools import (  # noqa: F401
    filesystem,
    http,
    python_exec,
    rag_tools,
    sql,
    web_search,
)
from agentic_studio.agents.tools.registry import REGISTRY, ToolRegistry, infer_schema, tool

SAFE_TOOL_NAMES = (
    "calculator",
    "web_search",
    "rag_search",
    "rag_answer",
    "graph_explore",
    "corpus_stats",
    "list_sources",
    "list_files",
    "read_file",
    "sql_query",
    "sql_schema",
    "http_request",
)

RESEARCH_TOOL_NAMES = ("web_search", "rag_search", "graph_explore", "list_sources")


def default_tools() -> list[ToolSpec]:
    """Tools an agent gets unless told otherwise: no writes, no code execution."""
    return REGISTRY.specs(allow=SAFE_TOOL_NAMES)


def research_tools() -> list[ToolSpec]:
    return REGISTRY.specs(allow=RESEARCH_TOOL_NAMES)


def all_tools() -> list[ToolSpec]:
    """Everything, including approval-gated tools like python_exec and write_file."""
    return REGISTRY.specs()


__all__ = [
    "REGISTRY",
    "RESEARCH_TOOL_NAMES",
    "SAFE_TOOL_NAMES",
    "ToolRegistry",
    "all_tools",
    "default_tools",
    "infer_schema",
    "research_tools",
    "tool",
]
