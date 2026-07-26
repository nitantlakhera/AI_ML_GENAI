"""Filesystem tools confined to a sandbox root.

Every path is resolved and checked against the root, so `../../etc/passwd` and
symlink escapes both fail. Writes require approval; reads and listings do not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_studio.agents.tools.registry import tool
from agentic_studio.settings import get_settings

MAX_READ_CHARS = 20000
MAX_WRITE_CHARS = 200000


def sandbox_root() -> Path:
    root = get_settings().paths.tool_sandbox
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def resolve_in_sandbox(relative_path: str) -> Path:
    """Resolve a path inside the sandbox or raise."""
    root = sandbox_root()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path '{relative_path}' escapes the sandbox root")
    return candidate


@tool(name="list_files", tags=("filesystem",))
def list_files(path: str = ".", pattern: str = "*") -> dict[str, Any]:
    """List files and folders inside the agent sandbox.

    Args:
        path: Directory relative to the sandbox root.
        pattern: Glob pattern to filter entries, for example '*.md'.
    """
    target = resolve_in_sandbox(path)
    if not target.exists():
        return {"ok": False, "error": f"{path} does not exist"}
    if target.is_file():
        return {"ok": True, "path": path, "entries": [{"name": target.name, "type": "file",
                                                       "bytes": target.stat().st_size}]}
    entries = []
    for item in sorted(target.glob(pattern)):
        entries.append(
            {
                "name": item.name,
                "type": "dir" if item.is_dir() else "file",
                "bytes": item.stat().st_size if item.is_file() else 0,
            }
        )
    return {"ok": True, "path": path, "count": len(entries), "entries": entries[:200]}


@tool(name="read_file", tags=("filesystem",))
def read_file(path: str, max_chars: int = MAX_READ_CHARS) -> dict[str, Any]:
    """Read a UTF-8 text file from the agent sandbox.

    Args:
        path: File path relative to the sandbox root.
        max_chars: Maximum number of characters to return.
    """
    target = resolve_in_sandbox(path)
    if not target.exists() or not target.is_file():
        return {"ok": False, "error": f"{path} is not a readable file"}
    text = target.read_text(encoding="utf-8", errors="replace")
    truncated = len(text) > max_chars
    return {
        "ok": True,
        "path": path,
        "truncated": truncated,
        "chars": len(text),
        "content": text[:max_chars],
    }


@tool(name="write_file", requires_approval=True, tags=("filesystem", "dangerous"))
def write_file(path: str, content: str, append: bool = False) -> dict[str, Any]:
    """Write a text file inside the agent sandbox.

    Args:
        path: File path relative to the sandbox root.
        content: Text to write.
        append: Append instead of overwriting.
    """
    if len(content) > MAX_WRITE_CHARS:
        return {"ok": False, "error": f"content exceeds {MAX_WRITE_CHARS} characters"}
    target = resolve_in_sandbox(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with target.open(mode, encoding="utf-8") as handle:
        handle.write(content)
    return {"ok": True, "path": path, "bytes": target.stat().st_size, "appended": append}


@tool(name="delete_file", requires_approval=True, tags=("filesystem", "dangerous"))
def delete_file(path: str) -> dict[str, Any]:
    """Delete a file inside the agent sandbox.

    Args:
        path: File path relative to the sandbox root.
    """
    target = resolve_in_sandbox(path)
    if not target.exists() or not target.is_file():
        return {"ok": False, "error": f"{path} is not a deletable file"}
    target.unlink()
    return {"ok": True, "path": path, "deleted": True}
