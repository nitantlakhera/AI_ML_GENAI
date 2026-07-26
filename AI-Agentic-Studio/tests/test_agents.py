"""Graph engine, checkpointing, interrupts, and the three agent architectures."""

from __future__ import annotations

import pytest

from agentic_studio.agents.checkpoint import MemoryCheckpointer, SqliteCheckpointer
from agentic_studio.agents.graph import (
    END,
    GraphError,
    Interrupt,
    StateGraph,
    add_messages,
    add_numbers,
)
from agentic_studio.agents.hitl import APPROVED, PENDING, get_approval_store
from agentic_studio.agents.planner import PlanExecuteAgent
from agentic_studio.agents.react import ToolCallingAgent
from agentic_studio.agents.supervisor import Specialist, SupervisorAgent, build_specialist
from agentic_studio.agents.tools import REGISTRY
from agentic_studio.core.types import Message
from agentic_studio.observability.metrics import METRICS

# -- graph engine -----------------------------------------------------------


def test_linear_graph_runs_nodes_in_order():
    graph = StateGraph(reducers={"trail": add_messages})
    graph.add_node("first", lambda state: {"trail": "first"})
    graph.add_node("second", lambda state: {"trail": "second"})
    graph.set_entry("first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)

    final = graph.compile().invoke({"trail": []})

    assert final["trail"] == ["first", "second"]
    assert final["__status__"] == "completed"


def test_conditional_edges_route_on_state():
    graph = StateGraph()
    graph.add_node("check", lambda state: {"value": state["value"] * 2})
    graph.add_node("small", lambda state: {"label": "small"})
    graph.add_node("large", lambda state: {"label": "large"})
    graph.set_entry("check")
    graph.add_conditional_edges(
        "check", lambda state: "large" if state["value"] > 10 else "small",
        {"small": "small", "large": "large"},
    )
    graph.add_edge("small", END)
    graph.add_edge("large", END)
    app = graph.compile()

    assert app.invoke({"value": 2})["label"] == "small"
    assert app.invoke({"value": 9})["label"] == "large"


def test_loop_is_bounded_by_max_steps():
    graph = StateGraph(reducers={"count": add_numbers})
    graph.add_node("tick", lambda state: {"count": 1})
    graph.set_entry("tick")
    graph.add_edge("tick", "tick")

    final = graph.compile().invoke({"count": 0}, config={"max_steps": 5})

    assert final["count"] == 5
    assert final["__status__"] == "max_steps_exceeded"


def test_reducers_append_while_plain_keys_replace():
    graph = StateGraph(reducers={"log": add_messages})
    graph.add_node("a", lambda state: {"log": "one", "latest": "one"})
    graph.add_node("b", lambda state: {"log": "two", "latest": "two"})
    graph.set_entry("a")
    graph.add_edge("a", "b")

    final = graph.compile().invoke({"log": []})

    assert final["log"] == ["one", "two"]
    assert final["latest"] == "two"


def test_interrupt_pauses_and_resume_continues(checkpointer):
    def gate(state):
        if state.get("__resume__") is None:
            raise Interrupt({"reason": "need_input", "question": "continue?"})
        return {"decision": state["__resume__"]}

    graph = StateGraph()
    graph.add_node("gate", gate)
    graph.add_node("finish", lambda state: {"done": True})
    graph.set_entry("gate")
    graph.add_edge("gate", "finish")
    app = graph.compile(checkpointer=checkpointer)

    paused = app.invoke({}, config={"thread_id": "t1"})
    assert paused["__status__"] == "interrupted"
    assert paused["__interrupt__"]["reason"] == "need_input"

    resumed = app.resume("t1", value="yes")
    assert resumed["decision"] == "yes"
    assert resumed["done"] is True
    assert resumed["__status__"] == "completed"


def test_sqlite_checkpointer_round_trips_messages(tmp_path):
    checkpointer = SqliteCheckpointer(path=tmp_path / "cp.sqlite3")
    checkpointer.save("t9", {"messages": [Message.user("hi")], "n": 3}, "next", 1)

    snapshot = checkpointer.load("t9")

    assert snapshot["next_node"] == "next"
    assert isinstance(snapshot["state"]["messages"][0], Message)
    assert snapshot["state"]["messages"][0].content == "hi"
    assert checkpointer.history("t9")[0]["step"] == 1
    assert checkpointer.delete("t9") is True


def test_compile_rejects_a_missing_entry_node():
    graph = StateGraph()
    graph.add_node("only", lambda state: None)
    graph.set_entry("missing")

    with pytest.raises(GraphError):
        graph.compile()


def test_mermaid_export_lists_nodes_and_branches():
    graph = StateGraph()
    graph.add_node("think", lambda state: None)
    graph.add_node("act", lambda state: None)
    graph.set_entry("think")
    graph.add_conditional_edges("think", lambda state: "act", {"act": "act", "end": END})
    graph.add_edge("act", "think")

    mermaid = graph.to_mermaid()

    assert "flowchart TD" in mermaid
    assert "think -->|act| act" in mermaid


# -- react agent ------------------------------------------------------------


def test_agent_calls_a_tool_then_answers(scripted):
    scripted.push({"tool_calls": [{"name": "calculator", "arguments": {"expression": "6*7"}}]})
    scripted.push("The answer is 42.")
    agent = ToolCallingAgent(tools=REGISTRY.specs(allow=["calculator"]), hitl=False)

    run = agent.run("what is 6 times 7", thread_id="calc")

    assert run.status == "completed"
    assert run.output == "The answer is 42."
    assert [step.node for step in run.steps] == ["think", "act", "think"]
    assert run.steps[1].results[0].output == "42"


def test_agent_runs_independent_tool_calls_in_parallel(scripted):
    scripted.push(
        {
            "tool_calls": [
                {"name": "calculator", "arguments": {"expression": "1+1"}},
                {"name": "calculator", "arguments": {"expression": "2+2"}},
            ]
        }
    )
    scripted.push("Both computed.")
    agent = ToolCallingAgent(tools=REGISTRY.specs(allow=["calculator"]), hitl=False)

    run = agent.run("compute both", thread_id="par")

    outputs = [result.output for result in run.steps[1].results]
    assert outputs == ["2", "4"], "results must stay in request order"


def test_tool_failure_is_reported_back_to_the_model(scripted):
    scripted.push({"tool_calls": [{"name": "does_not_exist", "arguments": {}}]})
    scripted.push("I could not use that tool.")
    agent = ToolCallingAgent(tools=REGISTRY.specs(allow=["calculator"]), hitl=False)

    run = agent.run("use a missing tool", thread_id="missing")

    result = run.steps[1].results[0]
    assert result.ok is False
    assert "not registered or not allowed" in (result.error or "")
    assert run.status == "completed"


def test_guardrails_block_a_tool_outside_the_allowlist(scripted):
    scripted.push({"tool_calls": [{"name": "write_file",
                                   "arguments": {"path": "x.txt", "content": "hi"}}]})
    scripted.push("Not permitted.")
    agent = ToolCallingAgent(tools=REGISTRY.specs(allow=["calculator"]), hitl=False)

    run = agent.run("write a file", thread_id="blocked")

    result = run.steps[1].results[0]
    assert result.ok is False
    assert "blocked by guardrails" in (result.error or "")


def test_approval_gate_pauses_then_resumes_on_approval(scripted):
    scripted.push({"tool_calls": [{"name": "python_exec",
                                   "arguments": {"code": "print(2+2)"}}]})
    scripted.push("The script printed 4.")
    agent = ToolCallingAgent(tools=REGISTRY.specs(allow=["python_exec"]), hitl=True)

    paused = agent.run("run some python", thread_id="gate")

    assert paused.status == "interrupted"
    assert paused.pending_approval["tool"] == "python_exec"
    pending = get_approval_store().pending("gate")
    assert len(pending) == 1
    assert pending[0]["status"] == PENDING

    request_id = paused.pending_approval["request_id"]
    resumed = agent.resume("gate", approved=True, request_id=request_id)

    assert get_approval_store().get(request_id)["status"] == APPROVED
    assert resumed.status == "completed"
    executed = [result for step in resumed.steps for result in step.results]
    assert any("4" in result.output for result in executed)


def test_rejected_approval_reports_the_refusal(scripted):
    scripted.push({"tool_calls": [{"name": "python_exec", "arguments": {"code": "print(1)"}}]})
    scripted.push("I was not allowed to run it.")
    agent = ToolCallingAgent(tools=REGISTRY.specs(allow=["python_exec"]), hitl=True)
    paused = agent.run("run python", thread_id="deny")

    resumed = agent.resume(
        "deny", approved=False, request_id=paused.pending_approval["request_id"], reason="unsafe"
    )

    failures = [r for step in resumed.steps for r in step.results if not r.ok]
    assert failures
    assert "not approved" in failures[0].error


def test_agent_stream_emits_one_event_per_node(scripted):
    scripted.push({"tool_calls": [{"name": "calculator", "arguments": {"expression": "3*3"}}]})
    scripted.push("Nine.")
    agent = ToolCallingAgent(tools=REGISTRY.specs(allow=["calculator"]), hitl=False)

    events = list(agent.stream("compute 3*3", thread_id="stream"))
    kinds = [event["type"] for event in events]

    assert kinds[0] == "think"
    assert "act" in kinds
    assert kinds[-1] == "done"
    assert events[-1]["output"] == "Nine."


def test_agent_records_metrics(scripted):
    scripted.push("Direct answer, no tools needed.")
    ToolCallingAgent(tools=[], hitl=False, name="metrics-agent").run("hello", thread_id="m")

    assert METRICS.counter("agent_runs", status="completed", agent="metrics-agent") == 1


def test_agent_without_tools_answers_directly(echo_router):
    run = ToolCallingAgent(tools=[], hitl=False).run("summarise hybrid retrieval", thread_id="none")

    assert run.status == "completed"
    assert run.output
    assert [step.node for step in run.steps] == ["think"]


# -- plan / execute / critique ----------------------------------------------


def test_plan_execute_agent_produces_a_plan_and_a_draft(echo_router):
    agent = PlanExecuteAgent(tools=REGISTRY.specs(allow=["rag_search"]), max_plan_steps=2,
                             max_revisions=0)

    run = agent.run("explain hybrid retrieval and reranking", thread_id="plan1")

    nodes = [step.node for step in run.steps]
    assert nodes[0] == "plan"
    assert any(node.startswith("execute") for node in nodes)
    assert "synthesize" in nodes
    assert "critique" in nodes
    assert run.output


# -- supervisor / multi-agent ------------------------------------------------


def test_supervisor_routes_to_a_specialist_and_finalizes(echo_router):
    specialist = build_specialist(
        name="researcher",
        description="finds facts in the corpus",
        tool_names=["rag_search"],
        system_prompt="You research.",
        router=echo_router,
    )
    team = SupervisorAgent(specialists=[specialist], max_handoffs=1, router=echo_router)

    run = team.run("what does BM25 catch", thread_id="team1")

    nodes = [step.node for step in run.steps]
    assert nodes[0] == "supervisor"
    assert "finalize" in nodes
    assert run.status == "completed"


def test_specialists_see_only_their_own_tools(echo_router):
    specialist = build_specialist(
        name="analyst", description="data", tool_names=["sql_query", "sql_schema"],
        system_prompt="You analyse.", router=echo_router,
    )

    assert isinstance(specialist, Specialist)
    assert specialist.agent.tool_names == ["sql_query", "sql_schema"]
