"""FastAPI application with Swagger UI for AI / ML / GenAI services."""

import uuid
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    AgentRequest,
    AgentResponse,
    AssistantRequest,
    AssistantResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestResponse,
    RAGRequest,
    RAGResponse,
    SourceDocument,
)
from chat.assistant import AIAssistant
from chat.bot import ChatBot
from config.settings import USE_API_LLM, VECTOR_DB_DIR

app = FastAPI(
    title="AI / ML / GenAI API",
    description=(
        "REST API for chatbot, AI assistant, RAG Q&A, and agents. "
        "Interactive docs: **/docs** (Swagger) and **/redoc** (ReDoc)."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_chat_sessions: dict[str, ChatBot] = {}
_assistant_sessions: dict[str, AIAssistant] = {}


def _session_id(provided: str | None) -> str:
    return provided or str(uuid.uuid4())


@lru_cache
def _get_rag_chain():
    from rag.chain import build_rag_chain
    from rag.embeddings import get_embeddings
    from rag.vector_store import load_vector_store

    embeddings = get_embeddings()
    vector_store = load_vector_store(VECTOR_DB_DIR, embeddings)
    return build_rag_chain(vector_store)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Check API and vector database status."""
    return HealthResponse(
        status="ok",
        vector_db_ready=VECTOR_DB_DIR.exists(),
        llm_mode="openai_api" if USE_API_LLM else "local_gguf",
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """Conversational chatbot with optional session memory."""
    session_id = _session_id(request.session_id)
    if session_id not in _chat_sessions:
        _chat_sessions[session_id] = ChatBot()
    try:
        reply = _chat_sessions[session_id].chat(request.message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc
    return ChatResponse(reply=reply, session_id=session_id)


@app.post("/assistant", response_model=AssistantResponse, tags=["Assistant"])
def assistant(request: AssistantRequest):
    """AI assistant with optional RAG grounding."""
    session_id = _session_id(request.session_id)
    if session_id not in _assistant_sessions:
        _assistant_sessions[session_id] = AIAssistant(use_rag=request.use_rag)
    bot = _assistant_sessions[session_id]
    try:
        answer = bot.ask(request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Assistant failed: {exc}") from exc
    return AssistantResponse(
        answer=answer,
        session_id=session_id,
        rag_used=bool(bot._rag_chain),
    )


@app.post("/rag/query", response_model=RAGResponse, tags=["RAG"])
def rag_query(request: RAGRequest):
    """Question answering grounded in indexed documents."""
    if not VECTOR_DB_DIR.exists():
        raise HTTPException(
            status_code=400,
            detail="Vector DB not found. Run: uv run python ingest.py",
        )
    try:
        chain = _get_rag_chain()
        result: dict[str, Any] = chain.invoke({"query": request.question})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {exc}") from exc

    sources = []
    for doc in result.get("source_documents", []):
        preview = doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else "")
        sources.append(
            SourceDocument(
                source=doc.metadata.get("source", "unknown"),
                content_preview=preview,
            )
        )
    return RAGResponse(answer=result["result"], sources=sources)


@app.post("/agent", response_model=AgentResponse, tags=["Agent"])
def agent_run(request: AgentRequest):
    """Run an AI agent with tools. Best with USE_API_LLM=true."""
    from agents.executor import run_agent

    try:
        result = run_agent(request.task)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent failed: {exc}") from exc
    return AgentResponse(result=result)


@app.post("/ingest", response_model=IngestResponse, tags=["RAG"])
def ingest_documents():
    """Rebuild FAISS index from files in data/raw/."""
    from config.settings import DATA_RAW_DIR
    from rag.embeddings import get_embeddings
    from rag.loader import load_documents
    from rag.splitter import split_documents
    from rag.vector_store import build_vector_store

    _get_rag_chain.cache_clear()

    docs = load_documents(DATA_RAW_DIR)
    if not docs:
        raise HTTPException(
            status_code=400,
            detail=f"No documents found in {DATA_RAW_DIR}",
        )
    chunks = split_documents(docs)
    embeddings = get_embeddings()
    build_vector_store(chunks, embeddings, VECTOR_DB_DIR)
    return IngestResponse(
        status="ok",
        chunks_indexed=len(chunks),
        message=f"Indexed {len(chunks)} chunks into {VECTOR_DB_DIR}",
    )


@app.delete("/sessions/{session_id}", tags=["Chat"])
def clear_session(session_id: str):
    """Clear chat or assistant session memory."""
    removed = False
    if session_id in _chat_sessions:
        del _chat_sessions[session_id]
        removed = True
    if session_id in _assistant_sessions:
        del _assistant_sessions[session_id]
        removed = True
    if not removed:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "ok", "session_id": session_id}
