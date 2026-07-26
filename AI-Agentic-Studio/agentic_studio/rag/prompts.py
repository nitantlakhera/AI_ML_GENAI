"""Prompt templates for grounded answering.

The `[n]` context labels are the contract between the prompt, the citation
parser, and the faithfulness metric in the evaluation harness.
"""

from __future__ import annotations

from agentic_studio.core.types import Retrieved

RAG_SYSTEM = (
    "You are a precise research assistant. Answer strictly from the provided context.\n"
    "Rules:\n"
    "1. Cite every claim with the bracketed context number it came from, e.g. [1] or [2][3].\n"
    "2. If the context does not contain the answer, say so plainly and do not guess.\n"
    "3. Prefer quoting concrete figures, names, and definitions from the context.\n"
    "4. Be concise: answer the question asked, without preamble."
)

CONVERSATIONAL_SYSTEM = (
    RAG_SYSTEM + "\n5. Use the conversation history only to interpret the question, "
    "never as a source of facts."
)

AGENT_SYSTEM = (
    "You are an autonomous agent that completes tasks using tools.\n"
    "Rules:\n"
    "1. Call a tool whenever it gives you information you do not already have.\n"
    "2. Never invent tool output; use only what the tool returned.\n"
    "3. Stop calling tools as soon as you can answer, then give the final answer directly.\n"
    "4. If a tool fails twice, explain the failure instead of retrying forever."
)


def build_context_block(contexts: list[Retrieved], max_chars: int = 12000,
                        use_parent: bool = True) -> str:
    """Render retrieved chunks as numbered, source-attributed blocks."""
    parts: list[str] = []
    budget = max_chars
    for index, item in enumerate(contexts, start=1):
        text = item.context_text if use_parent else item.text
        header = f"[{index}] (source: {_describe(item)})"
        body = text.strip()
        if len(body) > budget:
            body = body[: max(0, budget - 40)].rstrip() + " ..."
        block = f"{header}\n{body}"
        parts.append(block)
        budget -= len(block)
        if budget <= 0:
            break
    return "\n\n".join(parts)


def _describe(item: Retrieved) -> str:
    metadata = item.chunk.metadata
    label = metadata.get("title") or metadata.get("filename") or item.chunk.source
    page = metadata.get("page")
    heading = metadata.get("heading")
    if page:
        return f"{label} p.{page}"
    if heading:
        return f"{label} > {heading}"
    return str(label)


def build_rag_prompt(question: str, contexts: list[Retrieved], max_chars: int = 12000) -> str:
    return (
        f"Context:\n{build_context_block(contexts, max_chars)}\n\n"
        f"Question: {question}\n\n"
        "Answer with citations:"
    )


NO_CONTEXT_ANSWER = (
    "I could not find anything relevant in the indexed documents. "
    "Try rephrasing the question, or ingest more sources with `studio ingest`."
)
