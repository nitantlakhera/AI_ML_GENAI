"""The tool-calling agent.

Built as a two-node graph so every feature comes from the engine rather than
bespoke loop code:

    think ──tool_calls──> act ──> think
      │
      └──no tool_calls──> END

What the graph gives us for free: a step budget, checkpointed state, resumable
interrupts for approvals, and a trace span per node. Tool calls in one turn run
in parallel, guardrails vet every call, and failures are fed back to the model
as tool output instead of crashing the run.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from agentic_studio.agents.checkpoint import BaseCheckpointer, SqliteCheckpointer
from agentic_studio.agents.graph import END, Interrupt, StateGraph, State, add_messages, append_list
from agentic_studio.agents.hitl import APPROVED, get_approval_store
from agentic_studio.agents.tools import default_tools
from agentic_studio.agents.tools.registry import REGISTRY, ToolRegistry
from agentic_studio.core.types import (
    AgentRun,
    AgentStep,
    Message,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
    new_id,
)
from agentic_studio.guardrails.policy import get_policy
from agentic_studio.observability.logs import get_logger
from agentic_studio.observability.metrics import METRICS
from agentic_studio.rag.prompts import AGENT_SYSTEM
from agentic_studio.settings import get_settings

logger = get_logger("agents.react")


class ToolCallingAgent:
    def __init__(
        self,
        tools: list[ToolSpec] | None = None,
        router: Any = None,
        registry: ToolRegistry | None = None,
        system_prompt: str = AGENT_SYSTEM,
        max_steps: int | None = None,
        checkpointer: BaseCheckpointer | None = None,
        hitl: bool | None = None,
        name: str = "agent",
    ):
        settings = get_settings().agent
        self.name = name
        self.registry = registry or REGISTRY
        self.tools = tools if tools is not None else default_tools()
        if settings.allowed_tools:
            allowed = set(settings.allowed_tools)
            self.tools = [t for t in self.tools if t.name in allowed]
        self.system_prompt = system_prompt
        self.max_steps = max_steps or settings.max_steps
        self.hitl = settings.hitl_enabled if hitl is None else hitl
        self._router = router
        self.checkpointer = checkpointer or SqliteCheckpointer()
        self.app = self._build().compile(checkpointer=self.checkpointer, max_steps=self.max_steps * 2)

    @property
    def router(self) -> Any:
        if self._router is None:
            from agentic_studio.llm.router import get_router

            self._router = get_router()
        return self._router

    @property
    def tool_names(self) -> list[str]:
        return [tool.name for tool in self.tools]

    # -- graph --------------------------------------------------------------

    def _build(self) -> StateGraph:
        graph = StateGraph(
            reducers={"messages": add_messages, "steps": append_list, "usage": _merge_usage}
        )
        graph.add_node("think", self._think)
        graph.add_node("act", self._act)
        graph.set_entry("think")
        graph.add_conditional_edges("think", _route_after_think, {"act": "act", "end": END})
        graph.add_edge("act", "think")
        return graph

    def _think(self, state: State) -> State:
        messages: list[Message] = state.get("messages", [])
        response = self.router.generate(messages, tools=self.tools or None)
        step_index = len(state.get("steps", [])) + 1
        step = AgentStep(
            index=step_index,
            node="think",
            thought=response.text,
            tool_calls=list(response.tool_calls),
        )
        return {
            "messages": [response.to_message()],
            "steps": [step],
            "usage": response.usage,
            "pending_calls": [c.to_dict() for c in response.tool_calls],
            "output": response.text or state.get("output", ""),
        }

    def _act(self, state: State) -> State:
        pending = [
            ToolCall(name=item["name"], arguments=item.get("arguments", {}), id=item.get("id", ""))
            for item in state.get("pending_calls", [])
        ]
        if not pending:
            return {"pending_calls": []}

        thread_id = state.get("thread_id", "default")
        decision = state.pop("__resume__", None)
        policy = get_policy()
        allowed = set(self.tool_names)

        approved_calls: list[ToolCall] = []
        results: list[ToolResult] = []

        for call in pending:
            if not self.registry.has(call.name):
                # Distinguish a hallucinated tool name from a policy refusal, so
                # the model gets an actionable message instead of a scolding.
                results.append(
                    ToolResult(
                        tool_call_id=call.id,
                        name=call.name,
                        output="",
                        ok=False,
                        error=f"tool '{call.name}' is not registered or not allowed",
                    )
                )
                continue

            verdict = policy.check_tool(call.name, call.arguments, allowed=allowed)
            if not verdict.allowed:
                results.append(
                    ToolResult(
                        tool_call_id=call.id,
                        name=call.name,
                        output="",
                        ok=False,
                        error=f"blocked by guardrails: {'; '.join(verdict.notes)}",
                    )
                )
                continue

            gate = self._gate(call, thread_id, decision)
            if gate is None:
                approved_calls.append(call)
            else:
                results.append(gate)

        executed = self.registry.run_many(approved_calls) if approved_calls else []
        results.extend(executed)
        ordered = _reorder(results, pending)

        step = AgentStep(
            index=len(state.get("steps", [])) + 1,
            node="act",
            tool_calls=pending,
            results=ordered,
        )
        return {
            "messages": [Message.tool(result) for result in ordered],
            "steps": [step],
            "pending_calls": [],
        }

    def _gate(self, call: ToolCall, thread_id: str, decision: Any) -> ToolResult | None:
        """Return None to execute, or a ToolResult explaining why it was not run."""
        spec = self.registry.tools.get(call.name)
        if not self.hitl or spec is None or not spec.requires_approval:
            return None

        if isinstance(decision, dict) and decision.get("request_id"):
            record = get_approval_store().get(decision["request_id"])
            if record and record["status"] == APPROVED:
                return None
            reason = (record or {}).get("reason") or decision.get("reason") or "rejected by human"
            return ToolResult(
                tool_call_id=call.id, name=call.name, output="", ok=False,
                error=f"tool call was not approved: {reason}",
            )

        request_id = get_approval_store().create(thread_id, call.name, call.arguments)
        raise Interrupt(
            {
                "reason": "tool_approval_required",
                "request_id": request_id,
                "thread_id": thread_id,
                "tool": call.name,
                "arguments": call.arguments,
            }
        )

    # -- public API ---------------------------------------------------------

    def run(
        self,
        task: str,
        thread_id: str | None = None,
        history: list[Message] | None = None,
    ) -> AgentRun:
        started = time.perf_counter()
        thread_id = thread_id or new_id("thread")
        policy = get_policy()
        task = policy.check_input(task).raise_if_blocked()

        messages: list[Message] = [Message.system(self._system_text())]
        if history:
            messages.extend(history)
        messages.append(Message.user(task))

        final = self.app.invoke(
            {"messages": messages, "steps": [], "task": task, "thread_id": thread_id,
             "usage": Usage()},
            config={"thread_id": thread_id, "max_steps": self.max_steps * 2},
        )
        return self._to_run(task, thread_id, final, started)

    def resume(self, thread_id: str, approved: bool, request_id: str, reason: str = "") -> AgentRun:
        """Continue a run that paused for approval."""
        started = time.perf_counter()
        get_approval_store().decide(request_id, approved, reason=reason)
        final = self.app.resume(
            thread_id,
            value={"request_id": request_id, "approved": approved, "reason": reason},
            config={"max_steps": self.max_steps * 2},
        )
        return self._to_run(final.get("task", ""), thread_id, final, started)

    def stream(
        self, task: str, thread_id: str | None = None, history: list[Message] | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield an event per node so a UI can show the agent thinking."""
        thread_id = thread_id or new_id("thread")
        task = get_policy().check_input(task).raise_if_blocked()

        messages: list[Message] = [Message.system(self._system_text())]
        if history:
            messages.extend(history)
        messages.append(Message.user(task))

        for node, state in self.app.stream(
            {"messages": messages, "steps": [], "task": task, "thread_id": thread_id, "usage": Usage()},
            config={"thread_id": thread_id, "max_steps": self.max_steps * 2},
        ):
            steps: list[AgentStep] = state.get("steps", [])
            latest = steps[-1] if steps else None
            if node == END:
                yield {
                    "type": "done",
                    "status": state.get("__status__", "completed"),
                    "output": _final_output(state),
                    "thread_id": thread_id,
                }
            elif state.get("__status__") == "interrupted":
                yield {"type": "interrupted", "approval": state.get("__interrupt__"),
                       "thread_id": thread_id}
            elif latest is not None:
                yield {"type": node, "step": latest.to_dict(), "thread_id": thread_id}

    def _system_text(self) -> str:
        if not self.tools:
            return self.system_prompt
        catalog = "\n".join(f"- {tool.name}: {tool.description}" for tool in self.tools)
        return f"{self.system_prompt}\n\nAvailable tools:\n{catalog}"

    def _to_run(self, task: str, thread_id: str, state: State, started: float) -> AgentRun:
        status = state.get("__status__", "completed")
        output = _final_output(state)
        if status == "completed":
            output = get_policy().check_output(output).text

        METRICS.incr("agent_runs", status=status, agent=self.name)
        METRICS.observe("agent_latency_ms", (time.perf_counter() - started) * 1000, agent=self.name)

        return AgentRun(
            task=task,
            output=output,
            steps=state.get("steps", []),
            usage=state.get("usage") or Usage(),
            latency_ms=(time.perf_counter() - started) * 1000,
            status=status,
            thread_id=thread_id,
            pending_approval=state.get("__interrupt__"),
        )


def _route_after_think(state: State) -> str:
    return "act" if state.get("pending_calls") else "end"


def _merge_usage(existing: Any, incoming: Any) -> Usage:
    base = existing or Usage()
    return base + (incoming or Usage())


def _reorder(results: list[ToolResult], calls: list[ToolCall]) -> list[ToolResult]:
    by_id = {result.tool_call_id: result for result in results}
    ordered = [by_id[call.id] for call in calls if call.id in by_id]
    ordered.extend(result for result in results if result.tool_call_id not in {c.id for c in calls})
    return ordered


def _final_output(state: State) -> str:
    messages: list[Message] = state.get("messages", [])
    for message in reversed(messages):
        if message.role == "assistant" and message.content:
            return message.content
    return state.get("output", "")


def build_agent(**kwargs: Any) -> ToolCallingAgent:
    return ToolCallingAgent(**kwargs)


def run_agent(task: str, **kwargs: Any) -> AgentRun:
    """One-shot convenience wrapper."""
    return ToolCallingAgent(**kwargs).run(task)
