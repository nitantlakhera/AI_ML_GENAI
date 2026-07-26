"""Rerankers - the highest-leverage upgrade to a basic RAG stack.

Retrieval optimises recall over a large candidate pool; reranking optimises
precision over the handful of chunks the LLM will actually read.

* `lexical`       - zero-dependency term coverage + proximity (default)
* `cross-encoder` - sentence-transformers cross-encoder (best quality/cost)
* `llm`           - ask the model to grade relevance (best quality, slowest)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentic_studio.core.types import Message, Retrieved
from agentic_studio.observability.logs import get_logger
from agentic_studio.observability.tracing import get_tracer
from agentic_studio.rag.lexical import tokenize
from agentic_studio.settings import get_settings

logger = get_logger("rag.rerank")


class BaseReranker(ABC):
    name = "base"

    @abstractmethod
    def score(self, query: str, results: list[Retrieved]) -> list[float]:
        ...

    def rerank(self, query: str, results: list[Retrieved], top_k: int | None = None) -> list[Retrieved]:
        if not results:
            return []
        with get_tracer().span("retriever.rerank", kind="retriever", reranker=self.name,
                               candidates=len(results)):
            scores = self.score(query, results)
            paired = sorted(zip(results, scores), key=lambda pair: -pair[1])
            output: list[Retrieved] = []
            for rank, (item, score) in enumerate(paired[: top_k or len(paired)], start=1):
                output.append(
                    Retrieved(chunk=item.chunk, score=float(score),
                              retriever=f"{item.retriever}|rerank:{self.name}", rank=rank)
                )
            return output


class NoOpReranker(BaseReranker):
    name = "none"

    def score(self, query: str, results: list[Retrieved]) -> list[float]:
        return [item.score for item in results]


class LexicalReranker(BaseReranker):
    """Coverage of query terms, boosted when those terms appear close together."""

    name = "lexical"

    def score(self, query: str, results: list[Retrieved]) -> list[float]:
        query_terms = tokenize(query)
        unique_terms = set(query_terms)
        if not unique_terms:
            return [item.score for item in results]

        scores: list[float] = []
        for item in results:
            tokens = tokenize(item.text)
            token_set = set(tokens)
            coverage = len(unique_terms & token_set) / len(unique_terms)
            positions = [i for i, token in enumerate(tokens) if token in unique_terms]
            proximity = 0.0
            if len(positions) > 1:
                spread = positions[-1] - positions[0] + 1
                proximity = len(positions) / spread
            length_penalty = min(1.0, 400 / max(len(item.text), 1))
            scores.append(0.65 * coverage + 0.25 * proximity + 0.10 * length_penalty)
        return scores


class CrossEncoderReranker(BaseReranker):
    """Query-document pair scoring with a cross-encoder model."""

    name = "cross-encoder"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or get_settings().retrieval.rerank_model
        self._model: Any = None
        self._fallback = LexicalReranker()

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, results: list[Retrieved]) -> list[float]:
        try:
            model = self._load()
        except Exception as exc:
            logger.warning("cross-encoder unavailable (%s); using lexical reranker", exc)
            return self._fallback.score(query, results)
        pairs = [(query, item.text) for item in results]
        return [float(value) for value in model.predict(pairs)]


class LLMReranker(BaseReranker):
    """Grade each candidate 0-10 for usefulness in answering the question."""

    name = "llm"

    def __init__(self, router: Any = None, batch_size: int = 8):
        self._router = router
        self.batch_size = batch_size
        self._fallback = LexicalReranker()

    def _get_router(self) -> Any:
        if self._router is None:
            from agentic_studio.llm.router import get_router

            self._router = get_router()
        return self._router

    def score(self, query: str, results: list[Retrieved]) -> list[float]:
        from agentic_studio.llm.structured import generate_structured

        scores: list[float] = []
        for start in range(0, len(results), self.batch_size):
            batch = results[start : start + self.batch_size]
            listing = "\n\n".join(
                f"[{index}] {item.text[:600]}" for index, item in enumerate(batch, start=1)
            )
            schema = {
                "type": "object",
                "properties": {
                    "scores": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "relevance": {"type": "number"},
                            },
                            "required": ["id", "relevance"],
                        },
                    }
                },
                "required": ["scores"],
            }
            prompt = (
                f"Question: {query}\n\nCandidate passages:\n{listing}\n\n"
                "Score every passage from 0 (useless) to 10 (fully answers the question)."
            )
            try:
                parsed = generate_structured(
                    [Message.user(prompt)], schema, router=self._get_router(), retries=1
                )
                by_id = {int(entry["id"]): float(entry["relevance"]) for entry in parsed["scores"]}
                if not any(by_id.values()):
                    raise ValueError("degenerate scores")
                scores.extend(by_id.get(index, 0.0) for index in range(1, len(batch) + 1))
            except Exception as exc:
                logger.warning("LLM reranker failed (%s); using lexical for this batch", exc)
                scores.extend(self._fallback.score(query, batch))
        return scores


def build_reranker(name: str | None = None) -> BaseReranker:
    key = (name or get_settings().retrieval.reranker).strip().lower()
    if key in {"none", "off", ""}:
        return NoOpReranker()
    if key in {"cross-encoder", "crossencoder", "ce"}:
        return CrossEncoderReranker()
    if key == "llm":
        return LLMReranker()
    if key != "lexical":
        logger.warning("unknown reranker %r; using lexical", key)
    return LexicalReranker()
