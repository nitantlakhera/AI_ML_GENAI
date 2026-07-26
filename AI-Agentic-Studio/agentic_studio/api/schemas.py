"""Request and response models. These are the OpenAPI contract."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    providers: list[dict[str, Any]]
    corpus: dict[str, Any]
    guardrails_enabled: bool
    auth_enabled: bool


class ProvidersResponse(BaseModel):
    providers: list[dict[str, Any]]


class IngestRequest(BaseModel):
    path: str | None = Field(default=None, description="Directory or file to ingest. Defaults to data/raw.")
    texts: list[str] | None = Field(default=None, description="Raw texts to ingest instead of files.")
    source: str = Field(default="inline", description="Source label used when ingesting raw texts.")


class IngestResponse(BaseModel):
    job_id: str | None = None
    status: str
    chunks_indexed: int = 0
    documents: int = 0
    detail: dict[str, Any] = Field(default_factory=dict)


class RagRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    metadata_filter: dict[str, Any] | None = Field(
        default=None,
        description="Metadata predicate, e.g. {\"filetype\": \"pdf\", \"page\": {\"$lte\": 10}}",
    )


class SourceModel(BaseModel):
    id: str
    source: str
    score: float
    retriever: str
    rank: int
    text: str


class RagResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceModel]
    queries_used: list[str]
    usage: dict[str, Any]
    latency_ms: float
    guardrail_notes: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    thread_id: str | None = None
    use_rag: bool = True
    metadata_filter: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    thread_id: str
    reply: str
    sources: list[SourceModel] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)


class ThreadSummary(BaseModel):
    thread_id: str
    title: str | None = None
    messages: int
    created_at: float
    updated_at: float


class AgentRequest(BaseModel):
    task: str = Field(min_length=1, max_length=8000)
    thread_id: str | None = None
    tools: list[str] | None = Field(default=None, description="Restrict the agent to these tools.")
    mode: str = Field(default="react", description="react | plan | team")
    max_steps: int | None = Field(default=None, ge=1, le=50)


class AgentResponse(BaseModel):
    task: str
    output: str
    status: str
    thread_id: str
    steps: list[dict[str, Any]]
    usage: dict[str, Any]
    latency_ms: float
    pending_approval: dict[str, Any] | None = None


class ApprovalDecision(BaseModel):
    approved: bool
    reason: str = ""
    thread_id: str
    resume: bool = Field(default=True, description="Resume the paused agent run immediately.")


class ToolInfo(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    requires_approval: bool
    tags: list[str]


class EvalRequest(BaseModel):
    dataset: str | None = Field(default=None, description="Path to a JSONL golden set.")
    label: str = "api"
    compare_baseline: bool = False
    use_judge: bool = False


class JobStatus(BaseModel):
    job_id: str
    kind: str
    status: str
    created_at: float
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class ErrorResponse(BaseModel):
    detail: str
    rule: str | None = None
