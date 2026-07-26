"""Streamlit playground.

Six tabs, each mapping to one capability of the studio: streaming RAG chat,
retrieval inspection, agent runs with a live approval queue, ingestion,
evaluation, and observability.

Run with:  studio ui       (or: streamlit run agentic_studio/ui/app.py)
"""

from __future__ import annotations

import json
import time

import streamlit as st

from agentic_studio import __version__
from agentic_studio.agents.hitl import get_approval_store
from agentic_studio.agents.planner import PlanExecuteAgent
from agentic_studio.agents.react import ToolCallingAgent
from agentic_studio.agents.supervisor import SupervisorAgent
from agentic_studio.agents.tools import REGISTRY, default_tools
from agentic_studio.core.types import new_id
from agentic_studio.llm.router import get_router
from agentic_studio.observability.metrics import METRICS
from agentic_studio.observability.tracing import get_tracer
from agentic_studio.rag.conversational import ConversationalRag
from agentic_studio.rag.pipeline import get_pipeline
from agentic_studio.settings import get_settings

st.set_page_config(page_title="AI Agentic Studio", page_icon="◆", layout="wide")


@st.cache_resource
def pipeline():
    return get_pipeline()


@st.cache_resource
def chat_engine():
    return ConversationalRag(pipeline=pipeline())


def sidebar() -> None:
    settings = get_settings()
    st.sidebar.title("AI Agentic Studio")
    st.sidebar.caption(f"v{__version__}")

    with st.sidebar.expander("Model routing", expanded=True):
        for provider in get_router().describe():
            mark = "online" if provider["available"] else "unavailable"
            st.write(f"**{provider['provider']}** · {provider['model']} · {mark}")

    with st.sidebar.expander("Corpus"):
        st.json(pipeline().stats(), expanded=False)

    with st.sidebar.expander("Retrieval settings"):
        st.write(
            {
                "hybrid": settings.retrieval.hybrid_enabled,
                "graph_rag": settings.retrieval.graph_rag_enabled,
                "reranker": settings.retrieval.reranker,
                "query_transform": settings.retrieval.query_transform,
                "top_k": settings.retrieval.top_k,
            }
        )

    pending = get_approval_store().pending()
    if pending:
        st.sidebar.warning(f"{len(pending)} tool call(s) awaiting approval")


def tab_chat() -> None:
    st.subheader("Conversational RAG")
    st.caption("Persistent thread, follow-up rewriting, streamed answer with sources.")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = new_id("thread")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    columns = st.columns([3, 1])
    columns[0].code(st.session_state.thread_id, language=None)
    if columns[1].button("New thread", use_container_width=True):
        st.session_state.thread_id = new_id("thread")
        st.session_state.messages = []
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander(f"{len(message['sources'])} sources"):
                    for source in message["sources"]:
                        st.write(f"**[{source['rank']}]** {source['source']} · {source['score']:.3f}")
                        st.caption(source["text"][:400])

    question = st.chat_input("Ask about your documents")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        sources: list[dict] = []
        collected: list[str] = []
        for event in chat_engine().stream(st.session_state.thread_id, question):
            if event["type"] == "sources":
                sources = event["sources"]
            elif event["type"] == "token":
                collected.append(event["text"])
                placeholder.markdown("".join(collected) + "▌")
        answer = "".join(collected)
        placeholder.markdown(answer)
        if sources:
            with st.expander(f"{len(sources)} sources"):
                for source in sources:
                    st.write(f"**[{source['rank']}]** {source['source']} · {source['score']:.3f}")
                    st.caption(source["text"][:400])

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})


def tab_retrieval() -> None:
    st.subheader("Retrieval inspector")
    st.caption("See exactly which retriever surfaced each chunk, and how reranking reordered them.")

    query = st.text_input("Query", key="retrieval_query")
    top_k = st.slider("Top K", 1, 20, get_settings().retrieval.top_k)
    source_filter = st.text_input("Filter by source contains (optional)")

    if not st.button("Search", type="primary") or not query:
        return

    where = {"source": {"$contains": source_filter}} if source_filter else None
    started = time.perf_counter()
    contexts, queries = pipeline().retrieve(query, where=where, top_k=top_k)
    elapsed = (time.perf_counter() - started) * 1000

    st.write(f"**{len(contexts)}** chunks in {elapsed:.0f}ms")
    st.write("Queries searched:", queries)
    for context in contexts:
        with st.expander(
            f"[{context.rank}] {context.chunk.source} · score {context.score:.4f} · {context.retriever}"
        ):
            st.write(context.text)
            st.json(context.chunk.metadata, expanded=False)


def tab_agent() -> None:
    st.subheader("Agents")
    st.caption("react = tool loop · plan = plan/execute/critique · team = supervisor + specialists")

    mode = st.radio("Mode", ["react", "plan", "team"], horizontal=True)
    available = REGISTRY.names()
    chosen = st.multiselect(
        "Tools", available, default=[spec.name for spec in default_tools()]
    )
    task = st.text_area("Task", height=90)

    if st.button("Run agent", type="primary") and task:
        tools = REGISTRY.specs(allow=chosen)
        if mode == "plan":
            agent = PlanExecuteAgent(tools=tools)
        elif mode == "team":
            agent = SupervisorAgent()
        else:
            agent = ToolCallingAgent(tools=tools)

        with st.status("Agent working...", expanded=True) as status:
            if mode == "react":
                output = ""
                for event in agent.stream(task):
                    if event["type"] == "done":
                        output = event["output"]
                        status.update(label=f"Finished: {event['status']}", state="complete")
                    elif event["type"] == "interrupted":
                        st.warning("Paused for approval")
                        st.json(event["approval"])
                        status.update(label="Awaiting approval", state="error")
                    else:
                        step = event["step"]
                        tool_names = ", ".join(call["name"] for call in step["tool_calls"])
                        st.write(f"**{step['index']}. {step['node']}** {tool_names}")
                        if step["thought"]:
                            st.caption(step["thought"][:400])
                        for result in step["results"]:
                            st.code(result["output"][:800] or result.get("error", ""), language="json")
                if output:
                    st.markdown("### Answer")
                    st.markdown(output)
            else:
                run = agent.run(task)
                status.update(label=f"Finished: {run.status}", state="complete")
                for step in run.steps:
                    st.write(f"**{step.index}. {step.node}**")
                    if step.thought:
                        st.caption(step.thought[:500])
                st.markdown("### Answer")
                st.markdown(run.output)

    st.divider()
    st.markdown("#### Approval queue")
    pending = get_approval_store().pending()
    if not pending:
        st.caption("Nothing waiting.")
        return
    for record in pending:
        with st.container(border=True):
            st.write(f"**{record['tool']}** on thread `{record['thread_id']}`")
            st.json(record["arguments"])
            columns = st.columns(2)
            if columns[0].button("Approve", key=f"ok-{record['request_id']}", type="primary"):
                run = ToolCallingAgent().resume(
                    record["thread_id"], approved=True, request_id=record["request_id"]
                )
                st.success(run.output or run.status)
            if columns[1].button("Reject", key=f"no-{record['request_id']}"):
                run = ToolCallingAgent().resume(
                    record["thread_id"], approved=False, request_id=record["request_id"],
                    reason="rejected in UI",
                )
                st.info(run.output or run.status)


def tab_ingest() -> None:
    st.subheader("Ingestion")
    st.caption("Chunking is idempotent: re-ingesting a changed file replaces only its chunks.")

    settings = get_settings()
    st.write(f"Source directory: `{settings.paths.data_raw}`")

    uploaded = st.file_uploader(
        "Upload documents", accept_multiple_files=True, type=["txt", "md", "pdf", "json", "csv", "html"]
    )
    if uploaded and st.button("Ingest uploads", type="primary"):
        settings.paths.data_raw.mkdir(parents=True, exist_ok=True)
        for item in uploaded:
            (settings.paths.data_raw / item.name).write_bytes(item.getbuffer())
        from agentic_studio.rag.ingest import ingest_directory

        st.json(ingest_directory(pipeline=pipeline()))
        st.cache_resource.clear()

    if st.button("Re-index source directory"):
        from agentic_studio.rag.ingest import ingest_directory

        st.json(ingest_directory(pipeline=pipeline()))
        st.cache_resource.clear()

    pasted = st.text_area("Or paste text to index", height=140)
    if pasted and st.button("Ingest text"):
        from agentic_studio.rag.ingest import ingest_texts

        st.json(ingest_texts([pasted], source="pasted", pipeline=pipeline()))
        st.cache_resource.clear()


def tab_eval() -> None:
    st.subheader("Evaluation")
    st.caption("Score the pipeline on a golden set, or A/B it against a naive baseline.")

    from agentic_studio.evaluation.datasets import default_dataset_path, load_dataset, write_sample_dataset
    from agentic_studio.evaluation.runner import EvalRunner, compare_configs

    path = default_dataset_path()
    if not path.exists():
        st.info(f"No golden set yet at {path}")
        if st.button("Create starter golden set"):
            write_sample_dataset(path)
            st.rerun()
        return

    cases = load_dataset(path)
    st.write(f"**{len(cases)}** cases in `{path.name}`")
    compare = st.checkbox("Compare against naive baseline")

    if st.button("Run evaluation", type="primary"):
        with st.spinner("Scoring..."):
            if compare:
                outcome = compare_configs(cases)
                st.markdown("#### Baseline vs advanced")
                st.dataframe(
                    [
                        {
                            "metric": name,
                            "baseline": outcome["baseline"]["aggregate"].get(name, 0.0),
                            "advanced": outcome["advanced"]["aggregate"].get(name, 0.0),
                            "delta": delta,
                        }
                        for name, delta in outcome["delta"].items()
                    ],
                    use_container_width=True,
                )
            else:
                report = EvalRunner().run(cases, label="ui")
                st.markdown(report.to_markdown())


def tab_observability() -> None:
    st.subheader("Observability")
    st.caption("Counters, latency percentiles, token spend, and the most recent trace spans.")

    snapshot = METRICS.snapshot()
    columns = st.columns(4)
    columns[0].metric("LLM calls", int(sum(v for k, v in snapshot["counters"].items()
                                           if k.startswith("llm_calls"))))
    columns[1].metric("Tool calls", int(sum(v for k, v in snapshot["counters"].items()
                                            if k.startswith("tool_calls"))))
    columns[2].metric("Cache hits", int(sum(v for k, v in snapshot["counters"].items()
                                            if k.startswith("llm_cache_hits"))))
    columns[3].metric("Cost (USD)", f"{sum(v for k, v in snapshot['counters'].items() if k.startswith('llm_cost_usd')):.4f}")

    st.markdown("#### Counters")
    st.json(snapshot["counters"], expanded=False)
    st.markdown("#### Latency")
    st.json(snapshot["histograms"], expanded=False)

    st.markdown("#### Recent spans")
    spans = get_tracer().recent(60)
    if not spans:
        st.caption("No spans recorded yet.")
        return
    st.dataframe(
        [
            {
                "name": span["name"],
                "kind": span["kind"],
                "ms": span["duration_ms"],
                "status": span["status"],
                "trace": span["trace_id"][:8],
            }
            for span in reversed(spans)
        ],
        use_container_width=True,
        height=320,
    )
    with st.expander("Raw spans"):
        st.code(json.dumps(spans[-10:], indent=2), language="json")


def main() -> None:
    sidebar()
    st.title("AI Agentic Studio")
    tabs = st.tabs(["Chat", "Retrieval", "Agents", "Ingest", "Evaluation", "Observability"])
    with tabs[0]:
        tab_chat()
    with tabs[1]:
        tab_retrieval()
    with tabs[2]:
        tab_agent()
    with tabs[3]:
        tab_ingest()
    with tabs[4]:
        tab_eval()
    with tabs[5]:
        tab_observability()


main()
