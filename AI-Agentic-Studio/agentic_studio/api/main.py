"""The REST API.

Swagger UI at /docs. Everything the studio can do is reachable here: retrieval,
conversational RAG, three agent modes, streaming variants, human approvals,
ingestion and evaluation as background jobs, plus health, metrics, and traces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Path as PathParam, Query
from fastapi.middleware.cors import CORSMiddleware

from agentic_studio import __version__
from agentic_studio.agents.hitl import get_approval_store
from agentic_studio.agents.planner import PlanExecuteAgent
from agentic_studio.agents.react import ToolCallingAgent
from agentic_studio.agents.supervisor import SupervisorAgent
from agentic_studio.agents.tools import REGISTRY, default_tools
from agentic_studio.api.jobs import get_job_store
from agentic_studio.api.schemas import (
    AgentRequest,
    AgentResponse,
    ApprovalDecision,
    ChatRequest,
    ChatResponse,
    EvalRequest,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    JobStatus,
    ProvidersResponse,
    RagRequest,
    RagResponse,
    SourceModel,
    ThreadSummary,
    ToolInfo,
)
from agentic_studio.api.security import rate_limit_middleware, require_api_key
from agentic_studio.api.streaming import sse_response
from agentic_studio.core.errors import GuardrailBlocked, StudioError
from agentic_studio.core.types import new_id
from agentic_studio.llm.router import get_router
from agentic_studio.memory.store import get_store
from agentic_studio.observability.metrics import METRICS
from agentic_studio.observability.tracing import get_tracer
from agentic_studio.rag.conversational import ConversationalRag
from agentic_studio.rag.pipeline import get_pipeline
from agentic_studio.settings import get_settings

app = FastAPI(
    title="AI Agentic Studio API",
    description=(
        "Generative + Agentic AI platform: advanced RAG (hybrid retrieval, reranking, "
        "graph RAG), stateful agents with human-in-the-loop, MCP tool bridging, guardrails, "
        "observability, and evaluation.\n\n"
        "Interactive docs: **/docs** (Swagger) and **/redoc**."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

_settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(rate_limit_middleware)

Auth = Depends(require_api_key)


@app.exception_handler(GuardrailBlocked)
async def _guardrail_handler(_request: Any, exc: GuardrailBlocked):  # type: ignore[no-untyped-def]
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=422, content={"detail": str(exc), "rule": exc.rule})


@app.exception_handler(StudioError)
async def _studio_handler(_request: Any, exc: StudioError):  # type: ignore[no-untyped-def]
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=500, content={"detail": str(exc)})


# -- system ------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Service health plus the active provider chain and corpus size."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=__version__,
        providers=get_router().describe(),
        corpus=get_pipeline().stats(),
        guardrails_enabled=settings.guardrails.enabled,
        auth_enabled=settings.api.auth_enabled,
    )


@app.get("/providers", response_model=ProvidersResponse, tags=["System"])
def providers() -> ProvidersResponse:
    """The LLM routing chain, in failover order."""
    return ProvidersResponse(providers=get_router().describe())


@app.get("/metrics", tags=["System"])
def metrics() -> dict[str, Any]:
    """Counters and latency histograms for calls, tools, cache, and guardrails."""
    return METRICS.snapshot()


@app.get("/traces", tags=["System"])
def traces(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
    """Recent trace spans, newest last. Use trace_id to reconstruct one request."""
    return {"spans": get_tracer().recent(limit)}


@app.get("/tools", response_model=list[ToolInfo], tags=["Agents"])
def list_tools() -> list[ToolInfo]:
    """Every registered tool, including any bridged in from MCP servers."""
    return [ToolInfo(**info) for info in REGISTRY.describe()]


# -- ingestion ---------------------------------------------------------------


@app.post("/ingest", response_model=IngestResponse, tags=["RAG"])
def ingest(
    request: IngestRequest,
    background: BackgroundTasks,
    _: str = Auth,
) -> IngestResponse:
    """Index documents. Directory ingestion runs as a background job."""
    from agentic_studio.rag.ingest import ingest_directory, ingest_file, ingest_texts

    if request.texts:
        stats = ingest_texts(request.texts, source=request.source)
        return IngestResponse(
            status="completed",
            chunks_indexed=stats.get("chunks_indexed", 0),
            documents=stats.get("documents", 0),
            detail=stats,
        )

    target = Path(request.path) if request.path else get_settings().paths.data_raw
    store = get_job_store()
    job_id = store.create("ingest")

    def work() -> dict[str, Any]:
        if target.is_file():
            return ingest_file(target)
        return ingest_directory(target)

    background.add_task(store.run, job_id, work)
    return IngestResponse(job_id=job_id, status="queued", detail={"path": str(target)})


# -- retrieval ---------------------------------------------------------------


@app.post("/rag/query", response_model=RagResponse, tags=["RAG"])
def rag_query(request: RagRequest, _: str = Auth) -> RagResponse:
    """Answer a question from the indexed corpus, with citations and sources."""
    result = get_pipeline().answer(
        request.question, where=request.metadata_filter, top_k=request.top_k
    )
    return RagResponse(
        question=result.question,
        answer=result.answer,
        sources=[SourceModel(**context.to_dict()) for context in result.contexts],
        queries_used=result.queries_used,
        usage=result.usage.to_dict(),
        latency_ms=result.latency_ms,
        guardrail_notes=result.guardrail_notes,
    )


@app.post("/rag/stream", tags=["RAG"])
def rag_stream(request: RagRequest, _: str = Auth):  # type: ignore[no-untyped-def]
    """Same as /rag/query but streamed: sources first, then answer tokens."""
    events = get_pipeline().stream_answer(request.question, where=request.metadata_filter)
    return sse_response(events)


@app.post("/rag/search", tags=["RAG"])
def rag_search(request: RagRequest, _: str = Auth) -> dict[str, Any]:
    """Retrieval only - inspect what the pipeline would feed the model."""
    contexts, queries = get_pipeline().retrieve(
        request.question, where=request.metadata_filter, top_k=request.top_k
    )
    return {
        "question": request.question,
        "queries_used": queries,
        "count": len(contexts),
        "sources": [context.to_dict() for context in contexts],
    }


@app.delete("/rag/documents", tags=["RAG"])
def delete_documents(
    source_contains: str = Query(min_length=1, description="Delete chunks whose source contains this."),
    _: str = Auth,
) -> dict[str, Any]:
    """Remove indexed chunks by source, without rebuilding the whole index."""
    removed = get_pipeline().delete(where={"source": {"$contains": source_contains}})
    return {"removed": removed, "source_contains": source_contains}


# -- chat --------------------------------------------------------------------


def _chat_engine() -> ConversationalRag:
    return ConversationalRag()


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest, _: str = Auth) -> ChatResponse:
    """Multi-turn chat with persistent memory and optional RAG grounding."""
    thread_id = request.thread_id or new_id("thread")
    if request.use_rag:
        result = _chat_engine().ask(thread_id, request.message, where=request.metadata_filter)
        return ChatResponse(
            thread_id=thread_id,
            reply=result.answer,
            sources=[SourceModel(**context.to_dict()) for context in result.contexts],
            usage=result.usage.to_dict(),
        )

    from agentic_studio.core.types import Message
    from agentic_studio.memory.summarizing import SummarizingMemory

    memory = SummarizingMemory()
    history = memory.load(thread_id)
    response = get_router().generate(
        [Message.system("You are a helpful assistant."), *history, Message.user(request.message)]
    )
    memory.append(thread_id, Message.user(request.message))
    memory.append(thread_id, Message.assistant(response.text))
    return ChatResponse(thread_id=thread_id, reply=response.text, usage=response.usage.to_dict())


@app.post("/chat/stream", tags=["Chat"])
def chat_stream(request: ChatRequest, _: str = Auth):  # type: ignore[no-untyped-def]
    """Streamed conversational RAG over a persistent thread."""
    thread_id = request.thread_id or new_id("thread")
    events = _chat_engine().stream(thread_id, request.message, where=request.metadata_filter)
    return sse_response(events)


@app.get("/threads", response_model=list[ThreadSummary], tags=["Chat"])
def list_threads(limit: int = Query(default=50, ge=1, le=200), _: str = Auth) -> list[ThreadSummary]:
    """Conversation threads, most recently active first."""
    return [ThreadSummary(**row) for row in get_store().list_threads(limit)]


@app.get("/threads/{thread_id}", tags=["Chat"])
def get_thread(thread_id: str = PathParam(min_length=1), _: str = Auth) -> dict[str, Any]:
    """Full message history for one thread, plus its running summary."""
    messages = get_store().history(thread_id)
    if not messages:
        raise HTTPException(status_code=404, detail="thread not found")
    summary = get_store().get_summary(thread_id)
    return {
        "thread_id": thread_id,
        "messages": [message.to_dict() for message in messages],
        "summary": summary[0] if summary else None,
    }


@app.delete("/threads/{thread_id}", tags=["Chat"])
def delete_thread(thread_id: str = PathParam(min_length=1), _: str = Auth) -> dict[str, Any]:
    """Forget a thread: messages and summary."""
    if not get_store().delete_thread(thread_id):
        raise HTTPException(status_code=404, detail="thread not found")
    return {"deleted": True, "thread_id": thread_id}


# -- agents ------------------------------------------------------------------


def _build_agent(request: AgentRequest) -> Any:
    tools = REGISTRY.specs(allow=request.tools) if request.tools else default_tools()
    if request.mode == "plan":
        return PlanExecuteAgent(tools=tools)
    if request.mode == "team":
        return SupervisorAgent()
    return ToolCallingAgent(tools=tools, max_steps=request.max_steps)


@app.post("/agent", response_model=AgentResponse, tags=["Agents"])
def run_agent(request: AgentRequest, _: str = Auth) -> AgentResponse:
    """Run an agent. `mode` selects react (tool loop), plan (plan/critique), or team (multi-agent).

    A run that needs approval returns status `interrupted` with `pending_approval`;
    decide it at POST /agent/approvals/{request_id} to continue.
    """
    if request.mode not in {"react", "plan", "team"}:
        raise HTTPException(status_code=400, detail="mode must be react, plan, or team")

    run = _build_agent(request).run(request.task, thread_id=request.thread_id)
    return AgentResponse(**run.to_dict())


@app.post("/agent/stream", tags=["Agents"])
def stream_agent(request: AgentRequest, _: str = Auth):  # type: ignore[no-untyped-def]
    """Stream an agent's reasoning: one event per graph node."""
    if request.mode != "react":
        raise HTTPException(status_code=400, detail="streaming is available for mode=react")
    agent = ToolCallingAgent(
        tools=REGISTRY.specs(allow=request.tools) if request.tools else default_tools(),
        max_steps=request.max_steps,
    )
    return sse_response(agent.stream(request.task, thread_id=request.thread_id))


@app.get("/agent/approvals", tags=["Agents"])
def pending_approvals(
    thread_id: str | None = Query(default=None), _: str = Auth
) -> dict[str, Any]:
    """Tool calls waiting for a human decision."""
    return {"pending": get_approval_store().pending(thread_id)}


@app.post("/agent/approvals/{request_id}", response_model=AgentResponse, tags=["Agents"])
def decide_approval(
    decision: ApprovalDecision,
    request_id: str = PathParam(min_length=1),
    _: str = Auth,
) -> AgentResponse:
    """Approve or reject a paused tool call and resume the agent run."""
    record = get_approval_store().get(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail="approval request not found")

    agent = ToolCallingAgent()
    run = agent.resume(
        decision.thread_id, approved=decision.approved, request_id=request_id, reason=decision.reason
    )
    return AgentResponse(**run.to_dict())


# -- evaluation --------------------------------------------------------------


@app.post("/eval/run", response_model=JobStatus, tags=["Evaluation"])
def run_eval(request: EvalRequest, background: BackgroundTasks, _: str = Auth) -> JobStatus:
    """Score the RAG pipeline against a golden set. Runs as a background job."""
    from agentic_studio.evaluation.datasets import default_dataset_path, load_dataset, write_sample_dataset
    from agentic_studio.evaluation.judge import LLMJudge
    from agentic_studio.evaluation.runner import EvalRunner, compare_configs, write_report

    dataset_path = Path(request.dataset) if request.dataset else default_dataset_path()
    if not dataset_path.exists():
        dataset_path = write_sample_dataset(dataset_path)

    store = get_job_store()
    job_id = store.create("eval")

    def work() -> dict[str, Any]:
        cases = load_dataset(dataset_path)
        judge = LLMJudge() if request.use_judge else None
        if request.compare_baseline:
            return compare_configs(cases, judge=judge)
        report = EvalRunner(judge=judge).run(cases, label=request.label)
        paths = write_report(report)
        return {**report.to_dict(), "report_files": {k: str(v) for k, v in paths.items()}}

    background.add_task(store.run, job_id, work)
    return JobStatus(**store.get(job_id))  # type: ignore[arg-type]


# -- jobs --------------------------------------------------------------------


@app.get("/jobs", response_model=list[JobStatus], tags=["System"])
def list_jobs(limit: int = Query(default=25, ge=1, le=200), _: str = Auth) -> list[JobStatus]:
    """Recent background jobs."""
    return [JobStatus(**row) for row in get_job_store().list(limit)]


@app.get("/jobs/{job_id}", response_model=JobStatus, tags=["System"])
def get_job(job_id: str = PathParam(min_length=1), _: str = Auth) -> JobStatus:
    """Poll one background job for completion and its result."""
    record = get_job_store().get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatus(**record)
