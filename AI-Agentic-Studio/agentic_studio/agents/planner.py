"""Plan → Execute → Critique → Revise.

A single ReAct loop is greedy: it decides the next tool call from local context
and has no representation of the overall goal. For multi-part research tasks
that produces partial answers. This agent writes a plan first, executes each step
with its own tool budget, drafts an answer, then critiques and revises it.

    plan → execute ⇄ execute → synthesize → critique → (revise | END)
"""

from __future__ import annotations

import time
from typing import Any

from agentic_studio.agents.checkpoint import BaseCheckpointer, MemoryCheckpointer
from agentic_studio.agents.graph import END, StateGraph, State, append_list
from agentic_studio.agents.react import ToolCallingAgent
from agentic_studio.core.types import AgentRun, AgentStep, Message, ToolSpec, Usage
from agentic_studio.observability.logs import get_logger
from agentic_studio.observability.tracing import get_tracer

logger = get_logger("agents.planner")

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["goal"],
            },
        }
    },
    "required": ["steps"],
}

CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "missing": {"type": "string"},
    },
    "required": ["approved", "issues"],
}


class PlanExecuteAgent:
    def __init__(
        self,
        tools: list[ToolSpec] | None = None,
        router: Any = None,
        max_plan_steps: int = 4,
        max_revisions: int = 1,
        checkpointer: BaseCheckpointer | None = None,
        worker: ToolCallingAgent | None = None,
    ):
        self.max_plan_steps = max_plan_steps
        self.max_revisions = max_revisions
        self._router = router
        self.worker = worker or ToolCallingAgent(tools=tools, router=router, name="planner-worker",
                                                 max_steps=6)
        self.app = self._build().compile(
            checkpointer=checkpointer or MemoryCheckpointer(), max_steps=40
        )

    @property
    def router(self) -> Any:
        if self._router is None:
            from agentic_studio.llm.router import get_router

            self._router = get_router()
        return self._router

    def _build(self) -> StateGraph:
        graph = StateGraph(reducers={"findings": append_list, "steps": append_list,
                                     "usage": _merge_usage})
        graph.add_node("plan", self._plan)
        graph.add_node("execute", self._execute)
        graph.add_node("synthesize", self._synthesize)
        graph.add_node("critique", self._critique)
        graph.set_entry("plan")
        graph.add_edge("plan", "execute")
        graph.add_conditional_edges(
            "execute", _more_steps, {"execute": "execute", "synthesize": "synthesize"}
        )
        graph.add_edge("synthesize", "critique")
        graph.add_conditional_edges("critique", _needs_revision,
                                    {"synthesize": "synthesize", "end": END})
        return graph

    # -- nodes --------------------------------------------------------------

    def _plan(self, state: State) -> State:
        from agentic_studio.llm.structured import generate_structured

        task = state["task"]
        catalog = "\n".join(f"- {t.name}: {t.description}" for t in self.worker.tools)
        prompt = (
            f"Task: {task}\n\nAvailable tools:\n{catalog}\n\n"
            f"Write at most {self.max_plan_steps} concrete research steps that together "
            "complete the task. Each step must be independently executable with the tools above."
        )
        try:
            parsed = generate_structured(prompt, PLAN_SCHEMA, router=self.router, retries=1)
            goals = [step["goal"] for step in parsed.get("steps", []) if step.get("goal")]
        except Exception as exc:
            logger.warning("planning failed (%s); using the task as a single step", exc)
            goals = []

        plan = goals[: self.max_plan_steps] or [task]
        return {
            "plan": plan,
            "cursor": 0,
            "steps": [AgentStep(index=1, node="plan", thought="\n".join(f"- {g}" for g in plan))],
        }

    def _execute(self, state: State) -> State:
        plan: list[str] = state.get("plan", [])
        cursor = int(state.get("cursor", 0))
        if cursor >= len(plan):
            return {"cursor": cursor}

        goal = plan[cursor]
        with get_tracer().span("planner.step", kind="agent", step=cursor + 1, goal=goal[:80]):
            run = self.worker.run(goal, thread_id=f"{state.get('thread_id', 'plan')}::{cursor}")

        finding = {"goal": goal, "result": run.output, "tools": [
            result.name for step in run.steps for result in step.results
        ]}
        return {
            "cursor": cursor + 1,
            "findings": [finding],
            "usage": run.usage,
            "steps": [
                AgentStep(
                    index=len(state.get("steps", [])) + 1,
                    node=f"execute[{cursor + 1}]",
                    thought=f"{goal}\n-> {run.output[:500]}",
                    results=[result for step in run.steps for result in step.results],
                )
            ],
        }

    def _synthesize(self, state: State) -> State:
        findings = state.get("findings", [])
        critique = state.get("critique")
        transcript = "\n\n".join(
            f"Step {index}: {item['goal']}\nFinding: {item['result']}"
            for index, item in enumerate(findings, start=1)
        )
        instruction = (
            f"Task: {state['task']}\n\nResearch findings:\n{transcript}\n\n"
            "Write the final answer using only these findings. Be specific and concise."
        )
        if critique:
            issues = "; ".join(critique.get("issues", []))
            instruction += (
                f"\n\nA reviewer rejected the previous draft for these reasons: {issues}. "
                f"Missing: {critique.get('missing', 'n/a')}. Fix them in this version."
            )

        response = self.router.generate([Message.user(instruction)])
        return {
            "draft": response.text,
            "usage": response.usage,
            "steps": [
                AgentStep(index=len(state.get("steps", [])) + 1, node="synthesize",
                          thought=response.text[:500])
            ],
        }

    def _critique(self, state: State) -> State:
        from agentic_studio.llm.structured import generate_structured

        prompt = (
            f"Task: {state['task']}\n\nProposed answer:\n{state.get('draft', '')}\n\n"
            "Judge whether the answer fully and accurately completes the task. "
            "Set approved=false only if something material is wrong or missing."
        )
        try:
            critique = generate_structured(prompt, CRITIQUE_SCHEMA, router=self.router, retries=1)
        except Exception as exc:
            logger.warning("critique failed (%s); accepting the draft", exc)
            critique = {"approved": True, "issues": []}

        revisions = int(state.get("revisions", 0))
        approved = bool(critique.get("approved", True))
        return {
            "critique": critique if not approved else None,
            "approved": approved,
            "revisions": revisions + (0 if approved else 1),
            "steps": [
                AgentStep(
                    index=len(state.get("steps", [])) + 1,
                    node="critique",
                    thought=("approved" if approved else
                             "rejected: " + "; ".join(critique.get("issues", []))),
                )
            ],
            "max_revisions": self.max_revisions,
        }

    # -- public API ---------------------------------------------------------

    def run(self, task: str, thread_id: str | None = None) -> AgentRun:
        started = time.perf_counter()
        thread_id = thread_id or "plan-exec"
        final = self.app.invoke(
            {"task": task, "thread_id": thread_id, "findings": [], "steps": [], "usage": Usage(),
             "revisions": 0, "max_revisions": self.max_revisions},
            config={"thread_id": thread_id},
        )
        return AgentRun(
            task=task,
            output=final.get("draft", ""),
            steps=final.get("steps", []),
            usage=final.get("usage") or Usage(),
            latency_ms=(time.perf_counter() - started) * 1000,
            status=final.get("__status__", "completed"),
            thread_id=thread_id,
        )


def _more_steps(state: State) -> str:
    plan = state.get("plan", [])
    return "execute" if int(state.get("cursor", 0)) < len(plan) else "synthesize"


def _needs_revision(state: State) -> str:
    if state.get("approved", True):
        return "end"
    return "synthesize" if int(state.get("revisions", 0)) <= int(state.get("max_revisions", 1)) else "end"


def _merge_usage(existing: Any, incoming: Any) -> Usage:
    return (existing or Usage()) + (incoming or Usage())
