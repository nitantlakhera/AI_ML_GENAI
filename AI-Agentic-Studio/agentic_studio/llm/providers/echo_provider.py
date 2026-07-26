"""Deterministic offline provider.

This is what makes the whole studio runnable, testable, and demoable with zero
API keys. It is not a language model - it is a set of deterministic behaviours
that exercise every code path the real providers do:

* extractive answering over a retrieved context block (so RAG demos work)
* query-variant generation (so query transformation works)
* JSON synthesis from a supplied schema (so structured output works)
* keyword-driven tool calls (so the agent loop actually runs tools)

Swap `STUDIO_LLM_PROVIDERS=openai` to get a real model; no other code changes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

from agentic_studio.core.types import LLMResponse, Message, ToolCall, ToolSpec
from agentic_studio.llm.base import BaseProvider

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how",
    "i", "in", "is", "it", "of", "on", "or", "that", "the", "then", "there", "these", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "with", "you", "your",
}

_CONTEXT_BLOCK = re.compile(r"\[(\d+)\]\s*(?:\(source:[^)]*\))?\s*\n(.*?)(?=\n\[\d+\]|\Z)", re.S)
_JSON_SCHEMA_BLOCK = re.compile(r"```json-schema\s*(\{.*?\})\s*```", re.S)

_TOOL_HINTS: dict[str, tuple[str, ...]] = {
    "python_exec": ("calculate", "compute", "sum of", "average", "how many", "math", "arithmetic"),
    "web_search": ("search", "latest", "news", "who won", "current", "today", "look up", "web"),
    "rag_search": ("document", "documents", "my docs", "knowledge base", "handbook", "policy",
                   "according to", "in the corpus"),
    "read_file": ("read the file", "open the file", "file contents"),
    "list_files": ("list files", "what files", "directory listing"),
    "sql_query": ("select ", "sql", "database", "table"),
    "http_request": ("http", "api endpoint", "call the api", "rest api"),
}


class EchoProvider(BaseProvider):
    name = "echo"
    supports_tools = True
    supports_vision = True
    supports_streaming = True

    def __init__(self, model: str = "echo", **kwargs: Any):
        super().__init__(model=model or "echo", **kwargs)

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        prompt = _last_user_text(messages)
        already_used_tools = any(m.role == "tool" for m in messages)

        if tools and not already_used_tools:
            call = self._pick_tool(prompt, tools)
            if call is not None:
                return LLMResponse(
                    text="",
                    provider=self.name,
                    model=self.model,
                    tool_calls=[call],
                    finish_reason="tool_calls",
                    usage=self._usage(messages, ""),
                )

        text = self._answer(messages, prompt)
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            usage=self._usage(messages, text),
        )

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        text = self.generate(messages, **kwargs).text
        for token in re.findall(r"\S+\s*", text):
            yield token

    # -- behaviours ---------------------------------------------------------

    def _answer(self, messages: list[Message], prompt: str) -> str:
        full = "\n".join(m.content for m in messages if m.content)

        schema_match = _JSON_SCHEMA_BLOCK.search(full)
        if schema_match:
            return json.dumps(_synthesize_from_schema(json.loads(schema_match.group(1))), indent=2)

        if _wants_query_variants(full):
            return "\n".join(_query_variants(_after_question_label(prompt), count=3))

        tool_output = "\n".join(m.content for m in messages if m.role == "tool")
        contexts = _CONTEXT_BLOCK.findall(full)
        if contexts:
            # Score against the question alone; the surrounding prompt would
            # otherwise flood the term set and make every sentence look relevant.
            return _extractive_answer(_after_question_label(prompt), contexts)
        if tool_output:
            return f"Based on the tool results: {_first_sentences(tool_output, 2)}"

        return _fallback_answer(prompt)

    def _pick_tool(self, prompt: str, tools: list[ToolSpec]) -> ToolCall | None:
        lowered = prompt.lower()
        by_name = {tool.name: tool for tool in tools}
        for tool_name, hints in _TOOL_HINTS.items():
            if tool_name not in by_name:
                continue
            if any(hint in lowered for hint in hints):
                return ToolCall(name=tool_name, arguments=_default_arguments(by_name[tool_name], prompt))
        return None


def _default_arguments(tool: ToolSpec, prompt: str) -> dict[str, Any]:
    """Fill a tool's required parameters with something plausible from the prompt."""
    schema = tool.parameters or {}
    properties: dict[str, Any] = schema.get("properties", {})
    required: list[str] = schema.get("required", list(properties))
    arguments: dict[str, Any] = {}
    for key in required:
        spec = properties.get(key, {})
        kind = spec.get("type", "string")
        if kind == "string":
            if key in {"code", "expression"}:
                arguments[key] = _extract_expression(prompt)
            elif key in {"query", "question", "q", "text"}:
                arguments[key] = prompt
            else:
                arguments[key] = prompt
        elif kind == "integer":
            arguments[key] = int(spec.get("default", 5))
        elif kind == "number":
            arguments[key] = float(spec.get("default", 1))
        elif kind == "boolean":
            arguments[key] = bool(spec.get("default", False))
        elif kind == "array":
            arguments[key] = []
        else:
            arguments[key] = {}
    return arguments


def _extract_expression(prompt: str) -> str:
    match = re.search(r"[-+]?\d[\d\s.]*(?:[-+*/][\d\s.]+)+", prompt)
    if match:
        return f"print({match.group(0).strip()})"
    numbers = re.findall(r"\d+(?:\.\d+)?", prompt)
    if len(numbers) >= 2:
        return f"print({' + '.join(numbers)})"
    return "print('no expression found')"


def _last_user_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user" and message.content:
            return message.content
    return messages[-1].content if messages else ""


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in _STOPWORDS and len(t) > 1]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 2]


def _extractive_answer(question: str, contexts: list[tuple[str, str]]) -> str:
    """Pick the sentences that best cover the question and cite their sources."""
    query_terms = set(_tokens(question))
    scored: list[tuple[float, str, str]] = []
    for index, body in contexts:
        for sentence in _sentences(body):
            terms = set(_tokens(sentence))
            if not terms:
                continue
            overlap = len(query_terms & terms)
            score = overlap / (len(query_terms) or 1) + 0.01 * overlap
            scored.append((score, sentence, index))

    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = [item for item in scored[:3] if item[0] > 0]
    if not chosen:
        return (
            "The retrieved documents do not contain enough information to answer that. "
            "Try rephrasing the question or ingesting more sources."
        )

    used = sorted({item[2] for item in chosen}, key=int)
    body = " ".join(item[1].rstrip(".") + "." for item in chosen)
    citations = " ".join(f"[{index}]" for index in used)
    return f"{body} {citations}".strip()


def _after_question_label(text: str) -> str:
    """Isolate the actual question from an instruction-wrapped prompt."""
    matches = re.findall(r"Question:[ \t]*(.+)", text)
    return matches[-1].strip() if matches else text


def _wants_query_variants(text: str) -> bool:
    lowered = text.lower()
    return "alternative search queries" in lowered or "rewrite the question" in lowered


def _query_variants(question: str, count: int) -> list[str]:
    terms = _tokens(question)
    core = " ".join(terms) or question
    templates = [
        core,
        f"{core} explanation details",
        f"what is {core}",
        f"{core} overview summary",
        f"how does {core} work",
    ]
    seen: list[str] = []
    for candidate in templates:
        if candidate not in seen:
            seen.append(candidate)
        if len(seen) >= count:
            break
    return seen


def _first_sentences(text: str, count: int) -> str:
    return " ".join(_sentences(text)[:count]) or text[:200]


def _fallback_answer(prompt: str) -> str:
    topic = " ".join(_tokens(prompt)[:8]) or "your request"
    return (
        f"[echo provider] No model configured, so here is a deterministic reply about {topic}. "
        "Set STUDIO_LLM_PROVIDERS to openai, anthropic, gemini, ollama, or llamacpp for real generation."
    )


def _synthesize_from_schema(schema: dict[str, Any]) -> Any:
    """Build a minimal instance that satisfies a JSON schema."""
    kind = schema.get("type", "object")
    if "enum" in schema:
        return schema["enum"][0]
    if "default" in schema:
        return schema["default"]
    if kind == "object":
        properties: dict[str, Any] = schema.get("properties", {})
        required = schema.get("required", list(properties))
        return {key: _synthesize_from_schema(properties.get(key, {})) for key in required}
    if kind == "array":
        item_schema = schema.get("items", {"type": "string"})
        return [_synthesize_from_schema(item_schema)]
    if kind == "integer":
        return 0
    if kind == "number":
        return 0.0
    if kind == "boolean":
        return False
    if kind == "null":
        return None
    return ""
