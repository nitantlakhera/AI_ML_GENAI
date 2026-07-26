"""Multi-agent supervisor.

One agent holding twenty tools makes worse decisions than a router handing work
to focused specialists: each specialist sees a small tool surface and a narrow
system prompt, which is easier for the model and easier to evaluate.

    supervisor ──route──> researcher ──> supervisor ──> analyst ──> ... ──> END

The supervisor picks the next specialist (or FINISH) from the shared transcript,
with a hard cap on hand-offs so a pair of specialists cannot ping-pong forever.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from agentic_studio.agents.checkpoint import BaseCheckpointer, MemoryCheckpointer
from agentic_studio.agents.graph import END, StateGraph, State, append_list
from agentic_studio.agents.react import ToolCallingAgent
from agentic_studio.agents.tools import REGISTRY
from agentic_studio.core.types import AgentRun, AgentStep, Message, Usage
from agentic_studio.observability.logs import get_logger

logger = get_logger("agents.supervisor")

FINISH = "FINISH"


@dataclass
class Specialist:
    name: str
    description: str
    agent: ToolCallingAgent


def build_specialist(
    name: str,
    description: str,
    tool_names: list[str],
    system_prompt: str,
    router: Any = None,
    max_steps: int = 6,
) -> Specialist:
    agent = ToolCallingAgent(
        tools=REGISTRY.specs(allow=tool_names),
        router=router,
        system_prompt=system_prompt,
        max_steps=max_steps,
        hitl=False,
        name=name,
    )
    return Specialist(name=name, description=description, agent=agent)


def default_team(router: Any = None) -> list[Specialist]:
    """A small, opinionated team that covers research, data, and computation."""
    return [
        build_specialist(
            name="researcher",
            description="Finds facts in the document corpus and on the web.",
            tool_names=["rag_search", "rag_answer", "web_search", "graph_explore", "list_sources"],
            system_prompt=(
                "You are a research specialist. Gather evidence with your tools and report "
                "findings with sources. Never speculate beyond what the tools returned."
            ),
            router=router,
        ),
        build_specialist(
            name="analyst",
            description="Queries the database and interprets structured data.",
            tool_names=["sql_query", "sql_schema", "calculator"],
            system_prompt=(
                "You are a data analyst. Inspect the schema before querying, keep queries "
                "read-only, and state the numbers you found plainly."
            ),
            router=router,
        ),
        build_specialist(
            name="engineer",
            description="Runs calculations and inspects sandbox files.",
            tool_names=["calculator", "python_exec", "read_file", "list_files"],
            system_prompt=(
                "You are an engineer. Prefer the calculator for arithmetic and python_exec only "
                "when real computation is needed. Show the result, not the process."
            ),
            router=router,
        ),
    ]


class SupervisorAgent:
    def __init__(
        self,
        specialists: list[Specialist] | None = None,
        router: Any = None,
        max_handoffs: int = 4,
        checkpointer: BaseCheckpointer | None = None,
    ):
        self._router = router
        self.specialists = specialists or default_team(router)
        self.by_name = {s.name: s for s in self.specialists}
        self.max_handoffs = max_handoffs
        self.app = self._build().compile(
            checkpointer=checkpointer or MemoryCheckpointer(), max_steps=max_handoffs * 3 + 4
        )

    @property
    def router(self) -> Any:
        if self._router is None:
            from agentic_studio.llm.router import get_router

            self._router = get_router()
        return self._router

    def _build(self) -> StateGraph:
        graph = StateGraph(reducers={"transcript": append_list, "steps": append_list,
                                     "usage": _merge_usage})
        graph.add_node("supervisor", self._supervise)
        for specialist in self.specialists:
            graph.add_node(specialist.name, self._make_worker(specialist))
            graph.add_edge(specialist.name, "supervisor")
        graph.add_node("finalize", self._finalize)
        graph.set_entry("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._route,
            {**{s.name: s.name for s in self.specialists}, "finalize": "finalize"},
        )
        graph.add_edge("finalize", END)
        return graph

    # -- nodes --------------------------------------------------------------

    def _supervise(self, state: State) -> State:
        from agentic_studio.llm.structured import generate_structured

        handoffs = int(state.get("handoffs", 0))
        if handoffs >= self.max_handoffs:
            return {"next": FINISH, "reason": "handoff limit reached"}

        roster = "\n".join(f"- {s.name}: {s.description}" for s in self.specialists)
        transcript = _render(state.get("transcript", []))
        schema = {
            "type": "object",
            "properties": {
                "next": {"type": "string", "enum": [*self.by_name, FINISH]},
                "instruction": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["next"],
        }
        prompt = (
            f"Task: {state['task']}\n\nTeam:\n{roster}\n\n"
            f"Work so far:\n{transcript or '(nothing yet)'}\n\n"
            f"Choose the next specialist, or {FINISH} if the task is already answered. "
            "Give that specialist a specific instruction."
        )
        try:
            decision = generate_structured(prompt, schema, router=self.router, retries=1)
        except Exception as exc:
            logger.warning("supervisor routing failed (%s); finishing", exc)
            decision = {"next": FINISH, "reason": str(exc)}

        chosen = decision.get("next", FINISH)
        if chosen not in self.by_name:
            chosen = FINISH if handoffs > 0 or not self.specialists else self.specialists[0].name

        return {
            "next": chosen,
            "instruction": decision.get("instruction") or state["task"],
            "reason": decision.get("reason", ""),
            "steps": [
                AgentStep(
                    index=len(state.get("steps", [])) + 1,
                    node="supervisor",
                    thought=f"route -> {chosen}: {decision.get('reason', '')}",
                )
            ],
        }

    def _make_worker(self, specialist: Specialist):
        def worker(state: State) -> State:
            instruction = state.get("instruction") or state["task"]
            run = specialist.agent.run(
                instruction, thread_id=f"{state.get('thread_id', 'team')}::{specialist.name}"
            )
            return {
                "handoffs": int(state.get("handoffs", 0)) + 1,
                "transcript": [{"agent": specialist.name, "instruction": instruction,
                                "output": run.output}],
                "usage": run.usage,
                "steps": [
                    AgentStep(
                        index=len(state.get("steps", [])) + 1,
                        node=specialist.name,
                        thought=run.output[:500],
                        results=[r for step in run.steps for r in step.results],
                    )
                ],
            }

        return worker

    def _finalize(self, state: State) -> State:
        transcript = _render(state.get("transcript", []))
        if not transcript:
            return {"output": "No specialist produced a result."}
        response = self.router.generate(
            [
                Message.system("Combine the team's findings into one direct answer. No preamble."),
                Message.user(f"Task: {state['task']}\n\nFindings:\n{transcript}"),
            ]
        )
        return {
            "output": response.text,
            "usage": response.usage,
            "steps": [
                AgentStep(index=len(state.get("steps", [])) + 1, node="finalize",
                          thought=response.text[:500])
            ],
        }

    def _route(self, state: State) -> str:
        target = state.get("next", FINISH)
        return "finalize" if target == FINISH else target

    # -- public API ---------------------------------------------------------

    def run(self, task: str, thread_id: str | None = None) -> AgentRun:
        started = time.perf_counter()
        thread_id = thread_id or "team"
        final = self.app.invoke(
            {"task": task, "thread_id": thread_id, "transcript": [], "steps": [], "handoffs": 0,
             "usage": Usage()},
            config={"thread_id": thread_id},
        )
        return AgentRun(
            task=task,
            output=final.get("output", ""),
            steps=final.get("steps", []),
            usage=final.get("usage") or Usage(),
            latency_ms=(time.perf_counter() - started) * 1000,
            status=final.get("__status__", "completed"),
            thread_id=thread_id,
        )


def _render(transcript: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"[{item['agent']}] {item['output']}" for item in transcript)


def _merge_usage(existing: Any, incoming: Any) -> Usage:
    return (existing or Usage()) + (incoming or Usage())
