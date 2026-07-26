"""A small stateful graph engine for agent orchestration.

Sequential chains cannot express the shapes real agents need: loops (act until
done), branches (route by intent), retries, and pauses for human approval. This
module provides those with LangGraph-compatible semantics - nodes, conditional
edges, state reducers, checkpointing, and interrupts - in ~200 lines and with no
extra dependency.

    graph = StateGraph(reducers={"messages": add_messages})
    graph.add_node("think", think)
    graph.add_node("act", act)
    graph.set_entry("think")
    graph.add_conditional_edges("think", route, {"act": "act", "done": END})
    graph.add_edge("act", "think")
    app = graph.compile(checkpointer=SqliteCheckpointer())
    final = app.invoke({"messages": []}, config={"thread_id": "t1"})
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from agentic_studio.core.errors import StudioError
from agentic_studio.observability.logs import get_logger
from agentic_studio.observability.metrics import METRICS
from agentic_studio.observability.tracing import get_tracer

logger = get_logger("agents.graph")

State = dict[str, Any]
NodeFn = Callable[[State], "State | None"]
RouterFn = Callable[[State], str]

START = "__start__"
END = "__end__"
INTERRUPT_KEY = "__interrupt__"


class GraphError(StudioError):
    """The graph is malformed or exceeded its step budget."""


class Interrupt(Exception):
    """Raised by a node to pause execution and wait for outside input.

    The payload is handed to the caller (an approval UI, an API client) and the
    graph resumes at the same node once `resume(thread_id, value)` is called.
    """

    def __init__(self, payload: dict[str, Any]):
        super().__init__(f"interrupted: {payload.get('reason', 'awaiting input')}")
        self.payload = payload


def add_messages(existing: Any, incoming: Any) -> list[Any]:
    """Reducer that appends instead of replacing. Standard for `messages` state."""
    base = list(existing or [])
    if incoming is None:
        return base
    if isinstance(incoming, list):
        base.extend(incoming)
    else:
        base.append(incoming)
    return base


def append_list(existing: Any, incoming: Any) -> list[Any]:
    return add_messages(existing, incoming)


def add_numbers(existing: Any, incoming: Any) -> float:
    return (existing or 0) + (incoming or 0)


@dataclass
class StateGraph:
    reducers: dict[str, Callable[[Any, Any], Any]] = field(default_factory=dict)
    nodes: dict[str, NodeFn] = field(default_factory=dict)
    edges: dict[str, str] = field(default_factory=dict)
    branches: dict[str, tuple[RouterFn, dict[str, str]]] = field(default_factory=dict)
    entry: str | None = None

    def add_node(self, name: str, func: NodeFn) -> StateGraph:
        if name in {START, END}:
            raise GraphError(f"'{name}' is reserved")
        self.nodes[name] = func
        return self

    def add_edge(self, source: str, target: str) -> StateGraph:
        self.edges[source] = target
        return self

    def add_conditional_edges(
        self, source: str, router: RouterFn, mapping: dict[str, str] | None = None
    ) -> StateGraph:
        self.branches[source] = (router, mapping or {})
        return self

    def set_entry(self, name: str) -> StateGraph:
        self.entry = name
        return self

    def compile(self, checkpointer: Any = None, max_steps: int = 50) -> CompiledGraph:
        if not self.entry:
            raise GraphError("no entry node; call set_entry()")
        if self.entry not in self.nodes:
            raise GraphError(f"entry node '{self.entry}' is not registered")
        for source, target in self.edges.items():
            if source not in self.nodes:
                raise GraphError(f"edge from unknown node '{source}'")
            if target not in self.nodes and target != END:
                raise GraphError(f"edge to unknown node '{target}'")
        return CompiledGraph(self, checkpointer=checkpointer, max_steps=max_steps)

    def to_mermaid(self) -> str:
        """Render the graph so the docs and the code cannot drift apart."""
        lines = ["flowchart TD", f'    {START}(["start"]) --> {self.entry}']
        for name in self.nodes:
            lines.append(f'    {name}["{name}"]')
        for source, target in self.edges.items():
            lines.append(f"    {source} --> {_label(target)}")
        for source, (_, mapping) in self.branches.items():
            if mapping:
                for key, target in mapping.items():
                    lines.append(f"    {source} -->|{key}| {_label(target)}")
            else:
                lines.append(f"    {source} -.->|dynamic| {END}")
        lines.append(f'    {END}(["end"])')
        return "\n".join(lines)


def _label(target: str) -> str:
    return END if target == END else target


class CompiledGraph:
    def __init__(self, graph: StateGraph, checkpointer: Any = None, max_steps: int = 50):
        self.graph = graph
        self.checkpointer = checkpointer
        self.max_steps = max_steps

    # -- execution ----------------------------------------------------------

    def invoke(self, state: State, config: dict[str, Any] | None = None) -> State:
        final: State = dict(state)
        for _, snapshot in self.stream(state, config):
            final = snapshot
        return final

    def stream(
        self, state: State, config: dict[str, Any] | None = None
    ) -> Iterator[tuple[str, State]]:
        """Run the graph, yielding (node_name, state) after each node."""
        config = config or {}
        thread_id = config.get("thread_id", "default")
        max_steps = int(config.get("max_steps", self.max_steps))
        current = config.get("start_at") or self.graph.entry
        working: State = dict(state)
        step = int(working.pop("__step__", 0))

        with get_tracer().span("graph.run", kind="agent", thread=thread_id, entry=current) as span:
            while current and current != END:
                if step >= max_steps:
                    working["__status__"] = "max_steps_exceeded"
                    logger.warning("graph hit max_steps=%d on thread %s", max_steps, thread_id)
                    break

                node = self.graph.nodes.get(current)
                if node is None:
                    raise GraphError(f"unknown node '{current}'")

                step += 1
                with get_tracer().span(f"node.{current}", kind="agent", step=step):
                    try:
                        updates = node(working)
                    except Interrupt as interrupt:
                        working[INTERRUPT_KEY] = interrupt.payload
                        working["__status__"] = "interrupted"
                        self._save(thread_id, working, current, step)
                        span.set(status="interrupted", node=current)
                        METRICS.incr("graph_interrupts")
                        yield current, working
                        return

                working = self._merge(working, updates)
                yield current, dict(working)

                current = self._next(current, working)
                self._save(thread_id, working, current, step)

            if working.get("__status__") in (None, "running"):
                working["__status__"] = "completed"
            working["__step__"] = step
            span.set(status=working["__status__"], steps=step)
            METRICS.incr("graph_runs", status=working["__status__"])
            yield END, working

    def resume(self, thread_id: str, value: Any = None, config: dict[str, Any] | None = None) -> State:
        """Continue an interrupted run, injecting the outside decision."""
        if self.checkpointer is None:
            raise GraphError("resume requires a checkpointer")
        snapshot = self.checkpointer.load(thread_id)
        if snapshot is None:
            raise GraphError(f"no checkpoint for thread '{thread_id}'")

        state = snapshot["state"]
        state.pop(INTERRUPT_KEY, None)
        state.pop("__status__", None)
        state["__resume__"] = value
        merged = {**(config or {}), "thread_id": thread_id, "start_at": snapshot["next_node"]}
        return self.invoke(state, merged)

    def state(self, thread_id: str) -> State | None:
        if self.checkpointer is None:
            return None
        snapshot = self.checkpointer.load(thread_id)
        return snapshot["state"] if snapshot else None

    # -- internals ----------------------------------------------------------

    def _merge(self, state: State, updates: State | None) -> State:
        if not updates:
            return state
        merged = dict(state)
        for key, value in updates.items():
            reducer = self.graph.reducers.get(key)
            merged[key] = reducer(merged.get(key), value) if reducer else value
        return merged

    def _next(self, current: str, state: State) -> str:
        branch = self.graph.branches.get(current)
        if branch is not None:
            router, mapping = branch
            key = router(state)
            target = mapping.get(key, key) if mapping else key
            if target != END and target not in self.graph.nodes:
                raise GraphError(f"router from '{current}' returned unknown target '{target}'")
            return target
        return self.graph.edges.get(current, END)

    def _save(self, thread_id: str, state: State, next_node: str, step: int) -> None:
        if self.checkpointer is not None:
            self.checkpointer.save(thread_id, state, next_node, step)
