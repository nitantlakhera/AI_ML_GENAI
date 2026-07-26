from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    vector_db_ready: bool
    llm_mode: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, examples=["Hello!"])
    session_id: Optional[str] = Field(
        None,
        description="Optional session ID to keep conversation memory across requests",
    )


class ChatResponse(BaseModel):
    reply: str
    session_id: str


class AssistantRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["Summarize my documents"])
    use_rag: bool = Field(True, description="Ground answers in vector DB when available")
    session_id: Optional[str] = None


class AssistantResponse(BaseModel):
    answer: str
    session_id: str
    rag_used: bool


class RAGRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What does the sample document say?"])


class SourceDocument(BaseModel):
    source: str
    content_preview: str


class RAGResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]


class AgentRequest(BaseModel):
    task: str = Field(..., min_length=1, examples=["Calculate 15% of 240"])


class AgentResponse(BaseModel):
    result: str


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    message: str


class ErrorResponse(BaseModel):
    detail: str
