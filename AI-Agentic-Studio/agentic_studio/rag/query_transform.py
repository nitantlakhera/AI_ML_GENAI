"""Query transformation.

The user's phrasing is rarely the best retrieval query. Each strategy widens or
sharpens the search before any retriever runs:

* `rewrite`     - resolve pronouns and context from chat history into a standalone query
* `multi-query` - fan out into N paraphrases, then fuse the results
* `hyde`        - draft a hypothetical answer and search with that
* `decompose`   - break a multi-part question into sub-questions
"""

from __future__ import annotations

from typing import Any

from agentic_studio.core.types import Message
from agentic_studio.observability.logs import get_logger
from agentic_studio.observability.tracing import get_tracer
from agentic_studio.rag.lexical import tokenize
from agentic_studio.settings import get_settings

logger = get_logger("rag.query_transform")

STRATEGIES = ("none", "rewrite", "multi-query", "hyde", "decompose")


class QueryTransformer:
    def __init__(self, strategy: str | None = None, variants: int | None = None, router: Any = None):
        settings = get_settings().retrieval
        self.strategy = (strategy or settings.query_transform).strip().lower()
        self.variants = variants or settings.query_variants
        self._router = router

    def _get_router(self) -> Any:
        if self._router is None:
            from agentic_studio.llm.router import get_router

            self._router = get_router()
        return self._router

    def transform(self, question: str, history: list[Message] | None = None) -> list[str]:
        """Return the list of queries to search with; always includes the original."""
        if self.strategy in {"none", ""}:
            return [question]

        with get_tracer().span("rag.query_transform", kind="chain", strategy=self.strategy) as span:
            try:
                if self.strategy == "rewrite":
                    queries = [self.rewrite(question, history or [])]
                elif self.strategy == "multi-query":
                    queries = [question, *self.multi_query(question)]
                elif self.strategy == "hyde":
                    queries = [question, self.hyde(question)]
                elif self.strategy == "decompose":
                    queries = [question, *self.decompose(question)]
                else:
                    logger.warning("unknown query transform %r; passing through", self.strategy)
                    queries = [question]
            except Exception as exc:
                logger.warning("query transform failed (%s); using rule-based fallback", exc)
                queries = [question, *rule_based_variants(question, self.variants - 1)]

            unique = _dedupe(queries)[: max(1, self.variants)]
            span.set(queries=len(unique))
            return unique

    def rewrite(self, question: str, history: list[Message]) -> str:
        if not history:
            return question
        transcript = "\n".join(f"{m.role}: {m.content}" for m in history[-6:])
        prompt = (
            "Rewrite the question so it stands alone without the conversation. "
            "Resolve pronouns and implied subjects. Return only the rewritten question.\n\n"
            f"Conversation:\n{transcript}\n\nQuestion: {question}"
        )
        rewritten = self._get_router().complete(prompt).strip().splitlines()
        candidate = rewritten[0].strip(' "') if rewritten else question
        return candidate or question

    def multi_query(self, question: str) -> list[str]:
        count = max(1, self.variants - 1)
        prompt = (
            f"Generate {count} alternative search queries for the question below. "
            "Vary the vocabulary and specificity. One query per line, no numbering.\n\n"
            f"Question: {question}"
        )
        raw = self._get_router().complete(prompt)
        lines = [_clean(line) for line in raw.splitlines() if _clean(line)]
        return lines[:count] or rule_based_variants(question, count)

    def hyde(self, question: str) -> str:
        prompt = (
            "Write a short factual paragraph that would answer the question. "
            "It is used only as a retrieval probe, so plausibility matters more than accuracy.\n\n"
            f"Question: {question}"
        )
        draft = self._get_router().complete(prompt).strip()
        return draft or question

    def decompose(self, question: str) -> list[str]:
        count = max(1, self.variants - 1)
        prompt = (
            f"Break the question into at most {count} independent sub-questions that together "
            "answer it. One per line, no numbering. If it is already atomic, repeat it once.\n\n"
            f"Question: {question}"
        )
        raw = self._get_router().complete(prompt)
        lines = [_clean(line) for line in raw.splitlines() if _clean(line)]
        return lines[:count] or [question]


def rule_based_variants(question: str, count: int) -> list[str]:
    """Deterministic fallback used whenever the model is unavailable."""
    terms = tokenize(question)
    core = " ".join(terms)
    candidates = [
        core,
        f"{core} definition overview",
        f"how {core}",
        f"{core} example details",
    ]
    return _dedupe([c for c in candidates if c and c != question])[:count]


def _clean(line: str) -> str:
    stripped = line.strip()
    for prefix in ("- ", "* ", "• "):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
    if len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in {".", ")"}:
        stripped = stripped[2:]
    return stripped.strip(' "\'')


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item.strip())
    return output
