"""Provider-agnostic data types.

Everything in the studio speaks these types, so swapping an LLM provider, a
vector store, or a tool backend never leaks into business logic.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


def new_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class ToolCall:
    """A model's request to run one tool."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("call"))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class ToolResult:
    """The outcome of running one tool call."""

    tool_call_id: str
    name: str
    output: str
    ok: bool = True
    error: str | None = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "output": self.output,
            "ok": self.ok,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class Message:
    """One turn of a conversation, optionally carrying tool calls or images."""

    role: Role
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    images: list[str] = field(default_factory=list)

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str, images: list[str] | None = None) -> Message:
        return cls(role="user", content=content, images=images or [])

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, result: ToolResult) -> Message:
        return cls(
            role="tool",
            content=result.output if result.ok else f"ERROR: {result.error}",
            name=result.name,
            tool_call_id=result.tool_call_id,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            data["name"] = self.name
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            data["tool_calls"] = [c.to_dict() for c in self.tool_calls]
        if self.images:
            data["images"] = list(self.images)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", "") or "",
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id"),
            tool_calls=[
                ToolCall(name=c["name"], arguments=c.get("arguments", {}), id=c.get("id", new_id("call")))
                for c in data.get("tool_calls", [])
            ],
            images=list(data.get("images", [])),
        )


@dataclass
class ToolSpec:
    """A callable tool plus the JSON schema the model sees."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    requires_approval: bool = False
    tags: tuple[str, ...] = ()
    is_async: bool = False

    def json_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class Usage:
    """Token accounting for one model call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class LLMResponse:
    """A single completion, whatever produced it."""

    text: str = ""
    provider: str = ""
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_ms: float = 0.0
    cached: bool = False
    finish_reason: str = "stop"
    raw: Any = None

    def to_message(self) -> Message:
        return Message.assistant(self.text, self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "latency_ms": round(self.latency_ms, 2),
            "cached": self.cached,
            "finish_reason": self.finish_reason,
        }


@dataclass
class Document:
    """A source document before chunking."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("doc"))

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))


@dataclass
class Chunk:
    """A retrievable slice of a document."""

    text: str
    doc_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("chunk"))
    parent_text: str | None = None

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata,
            "parent_text": self.parent_text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(
            text=data["text"],
            doc_id=data.get("doc_id", ""),
            metadata=data.get("metadata", {}),
            id=data.get("id", new_id("chunk")),
            parent_text=data.get("parent_text"),
        )


@dataclass
class Retrieved:
    """A chunk with the score and the retriever that produced it."""

    chunk: Chunk
    score: float
    retriever: str = "dense"
    rank: int = 0

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def context_text(self) -> str:
        """Parent-document text when available, otherwise the chunk itself."""
        return self.chunk.parent_text or self.chunk.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.chunk.id,
            "source": self.chunk.source,
            "score": round(self.score, 6),
            "retriever": self.retriever,
            "rank": self.rank,
            "text": self.chunk.text,
        }


@dataclass
class RagAnswer:
    """The full result of a RAG query, including the audit trail."""

    question: str
    answer: str
    contexts: list[Retrieved] = field(default_factory=list)
    queries_used: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    guardrail_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": [c.to_dict() for c in self.contexts],
            "queries_used": self.queries_used,
            "usage": self.usage.to_dict(),
            "latency_ms": round(self.latency_ms, 2),
            "guardrail_notes": self.guardrail_notes,
        }


@dataclass
class AgentStep:
    """One observable step of an agent run."""

    index: int
    thought: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)
    node: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "node": self.node,
            "thought": self.thought,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class AgentRun:
    """The outcome of an agent invocation."""

    task: str
    output: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    status: str = "completed"
    thread_id: str = ""
    pending_approval: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "output": self.output,
            "status": self.status,
            "thread_id": self.thread_id,
            "steps": [s.to_dict() for s in self.steps],
            "usage": self.usage.to_dict(),
            "latency_ms": round(self.latency_ms, 2),
            "pending_approval": self.pending_approval,
        }
