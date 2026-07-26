"""Tool registry with schema inference, timeouts, retries, and parallel execution.

Schemas are derived from type hints and docstrings, so a tool is one decorated
function - there is no second place to keep in sync.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import types
import typing
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import Any, get_args, get_origin

from agentic_studio.core.errors import ToolNotFound
from agentic_studio.core.types import ToolCall, ToolResult, ToolSpec
from agentic_studio.observability.logs import get_logger
from agentic_studio.observability.metrics import METRICS
from agentic_studio.observability.tracing import get_tracer
from agentic_studio.settings import get_settings

logger = get_logger("agents.tools")

_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def infer_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Build a JSON schema for a function from its signature and docstring."""
    signature = inspect.signature(func)
    hints = typing.get_type_hints(func)
    descriptions = _param_docs(func.__doc__ or "")

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, parameter in signature.parameters.items():
        if name in {"self", "cls"} or parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        annotation = hints.get(name, str)
        schema = _annotation_schema(annotation)
        if name in descriptions:
            schema["description"] = descriptions[name]
        if parameter.default is not inspect.Parameter.empty:
            schema["default"] = parameter.default
        else:
            required.append(name)
        properties[name] = schema

    return {"type": "object", "properties": properties, "required": required}


def _annotation_schema(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    # `int | None` reports types.UnionType, `Optional[int]` reports typing.Union.
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        return _annotation_schema(args[0]) if args else {"type": "string"}
    if origin in {list, set, tuple}:
        args = get_args(annotation)
        return {"type": "array", "items": _annotation_schema(args[0]) if args else {"type": "string"}}
    if origin is dict:
        return {"type": "object"}
    if annotation in _JSON_TYPES:
        return {"type": _JSON_TYPES[annotation]}
    return {"type": "string"}


def _param_docs(docstring: str) -> dict[str, str]:
    """Read `name: description` lines from an Args/Parameters section."""
    documented: dict[str, str] = {}
    in_section = False
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped.lower() in {"args:", "arguments:", "parameters:"}:
            in_section = True
            continue
        if in_section:
            if not stripped or stripped.endswith(":") and " " not in stripped:
                break
            if ":" in stripped:
                name, _, description = stripped.partition(":")
                documented[name.strip()] = description.strip()
    return documented


def _summary(docstring: str | None) -> str:
    if not docstring:
        return ""
    lines: list[str] = []
    for line in docstring.strip().splitlines():
        if line.strip().lower() in {"args:", "arguments:", "parameters:", "returns:"}:
            break
        lines.append(line.strip())
    return " ".join(part for part in lines if part).strip()


@dataclass
class ToolRegistry:
    """Holds tool specs and executes tool calls safely."""

    tools: dict[str, ToolSpec]

    def __init__(self) -> None:
        self.tools = {}

    # -- registration -------------------------------------------------------

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self.tools:
            logger.info("replacing already-registered tool %s", spec.name)
        self.tools[spec.name] = spec
        return spec

    def tool(
        self,
        name: str | None = None,
        description: str | None = None,
        requires_approval: bool = False,
        tags: Iterable[str] = (),
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator: turn a plain function into a registered tool."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            spec = ToolSpec(
                name=name or func.__name__,
                description=description or _summary(func.__doc__) or func.__name__,
                parameters=infer_schema(func),
                func=func,
                requires_approval=requires_approval,
                tags=tuple(tags),
                is_async=inspect.iscoroutinefunction(func),
            )
            self.register(spec)
            func.tool_spec = spec  # type: ignore[attr-defined]
            return func

        return decorator

    def unregister(self, name: str) -> bool:
        return self.tools.pop(name, None) is not None

    # -- lookup -------------------------------------------------------------

    def get(self, name: str) -> ToolSpec:
        spec = self.tools.get(name)
        if spec is None:
            raise ToolNotFound(name)
        return spec

    def has(self, name: str) -> bool:
        return name in self.tools

    def specs(self, allow: Iterable[str] | None = None, tags: Iterable[str] | None = None) -> list[ToolSpec]:
        allowed = set(allow) if allow else None
        wanted = set(tags) if tags else None
        selected = []
        for spec in self.tools.values():
            if allowed is not None and spec.name not in allowed:
                continue
            if wanted is not None and not wanted & set(spec.tags):
                continue
            selected.append(spec)
        return sorted(selected, key=lambda s: s.name)

    def names(self) -> list[str]:
        return sorted(self.tools)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
                "requires_approval": spec.requires_approval,
                "tags": list(spec.tags),
            }
            for spec in self.specs()
        ]

    # -- execution ----------------------------------------------------------

    def run(
        self,
        call: ToolCall,
        timeout_s: float | None = None,
        retries: int | None = None,
    ) -> ToolResult:
        settings = get_settings().agent
        timeout_s = settings.tool_timeout_s if timeout_s is None else timeout_s
        retries = settings.tool_retries if retries is None else retries

        try:
            spec = self.get(call.name)
        except ToolNotFound as exc:
            METRICS.incr("tool_errors", tool=call.name)
            return ToolResult(tool_call_id=call.id, name=call.name, output="", ok=False, error=str(exc))

        last_error = ""
        for attempt in range(retries + 1):
            started = time.perf_counter()
            with get_tracer().span("tool.run", kind="tool", tool=spec.name, attempt=attempt + 1) as span:
                try:
                    value = _invoke(spec, call.arguments, timeout_s)
                    latency = (time.perf_counter() - started) * 1000
                    METRICS.incr("tool_calls", tool=spec.name)
                    METRICS.observe("tool_latency_ms", latency, tool=spec.name)
                    span.set(ok=True, latency_ms=round(latency, 2))
                    return ToolResult(
                        tool_call_id=call.id,
                        name=spec.name,
                        output=_stringify(value),
                        latency_ms=latency,
                    )
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    span.set(ok=False, error=last_error)
                    METRICS.incr("tool_errors", tool=spec.name)
                    if attempt < retries:
                        logger.info("tool %s failed (%s); retrying", spec.name, last_error)

        return ToolResult(tool_call_id=call.id, name=call.name, output="", ok=False, error=last_error)

    def run_many(
        self,
        calls: list[ToolCall],
        parallel: bool | None = None,
        timeout_s: float | None = None,
    ) -> list[ToolResult]:
        """Run independent tool calls concurrently, preserving request order."""
        settings = get_settings().agent
        parallel = settings.parallel_tools if parallel is None else parallel
        if not calls:
            return []
        if not parallel or len(calls) == 1:
            return [self.run(call, timeout_s=timeout_s) for call in calls]

        with ThreadPoolExecutor(max_workers=min(8, len(calls))) as pool:
            futures = [pool.submit(self.run, call, timeout_s) for call in calls]
            return [future.result() for future in futures]

    async def arun(self, call: ToolCall, timeout_s: float | None = None) -> ToolResult:
        return await asyncio.get_running_loop().run_in_executor(None, self.run, call, timeout_s)


def _invoke(spec: ToolSpec, arguments: dict[str, Any], timeout_s: float) -> Any:
    """Call a tool with a wall-clock timeout."""
    if spec.is_async:
        return asyncio.run(asyncio.wait_for(spec.func(**arguments), timeout=timeout_s))

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(spec.func, **arguments)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeout as exc:
            raise TimeoutError(f"tool '{spec.name}' exceeded {timeout_s}s") from exc


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, indent=2, default=str)
    except TypeError:
        return str(value)


REGISTRY = ToolRegistry()
tool = REGISTRY.tool
