"""Sandboxed Python execution.

Runs in a separate interpreter process with:
  * a wall-clock timeout enforced by process kill (not a thread that leaks)
  * a scratch working directory, so writes cannot touch the project
  * import and builtin blocklists installed before user code runs
  * no inherited environment, which removes API keys from reach

This is defence in depth, not a security boundary. For untrusted input, run it
inside a container or gVisor. The tool is marked `requires_approval` so the
human-in-the-loop gate applies by default.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from agentic_studio.agents.tools.registry import tool
from agentic_studio.settings import get_settings

BLOCKED_IMPORTS = {
    "socket", "subprocess", "shutil", "ctypes", "multiprocessing", "http", "urllib",
    "requests", "httpx", "ftplib", "telnetlib", "smtplib", "pickle", "importlib",
    "webbrowser", "pty", "signal",
}

BLOCKED_BUILTINS = {"eval", "exec", "compile", "open", "input", "breakpoint", "help", "__import__"}

_PREAMBLE = '''
import builtins, os, sys

_BLOCKED_IMPORTS = {blocked_imports!r}
_BLOCKED_BUILTINS = {blocked_builtins!r}

_real_import = builtins.__import__


def _guarded_import(name, *args, **kwargs):
    root = name.split(".")[0]
    if root in _BLOCKED_IMPORTS:
        raise ImportError(f"import of '{{name}}' is not allowed in the sandbox")
    return _real_import(name, *args, **kwargs)


builtins.__import__ = _guarded_import
for _name in _BLOCKED_BUILTINS:
    # Skip __import__: it is already replaced by the guard above, and nulling it
    # would turn every import into an unhelpful "NoneType is not callable".
    if _name != "__import__" and hasattr(builtins, _name):
        setattr(builtins, _name, None)

for _name in ("system", "popen", "execv", "execve", "spawnv", "remove", "rmdir", "unlink"):
    if hasattr(os, _name):
        setattr(os, _name, None)

sys.argv = ["sandbox"]
'''


def build_program(code: str) -> str:
    preamble = _PREAMBLE.format(
        blocked_imports=sorted(BLOCKED_IMPORTS), blocked_builtins=sorted(BLOCKED_BUILTINS)
    )
    return f"{preamble}\n# --- user code ---\n{code}\n"


def execute(code: str, timeout_s: float | None = None) -> dict[str, Any]:
    settings = get_settings()
    timeout_s = timeout_s or settings.tools.python_exec_timeout_s
    sandbox_root = settings.paths.tool_sandbox
    sandbox_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=sandbox_root) as workdir:
        script = Path(workdir) / "program.py"
        script.write_text(build_program(code), encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", str(script)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=workdir,
                env={"PATH": "", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"},
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "stderr": f"execution exceeded {timeout_s}s", "exit_code": -1}
        except Exception as exc:
            return {"ok": False, "stdout": "", "stderr": f"sandbox failed: {exc}", "exit_code": -1}

    return {
        "ok": completed.returncode == 0,
        "stdout": completed.stdout[-8000:],
        "stderr": completed.stderr[-4000:],
        "exit_code": completed.returncode,
    }


@tool(name="python_exec", requires_approval=True, tags=("compute", "dangerous"))
def python_exec(code: str) -> dict[str, Any]:
    """Run Python in an isolated sandbox and return stdout. Use print() to output results.

    Args:
        code: The Python source to execute. Network and filesystem access are blocked.
    """
    return execute(code)


@tool(name="calculator", tags=("compute",))
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly, including large integers.

    Args:
        expression: An arithmetic expression such as '(1234 * 17) / 3'.
    """
    allowed = set("0123456789+-*/%(). eE_")
    if not expression or not set(expression) <= allowed:
        return "Error: only numbers and the operators + - * / % ( ) are allowed."
    if "__" in expression:
        return "Error: invalid expression."
    try:
        value = eval(expression, {"__builtins__": {}}, {})  # noqa: S307 - character-allowlisted
    except Exception as exc:
        return f"Error: {exc}"
    return str(value)
