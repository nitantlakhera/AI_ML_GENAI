"""The advanced RAG pipeline.

    question
       -> guardrails (input)
       -> query transform (rewrite | multi-query | hyde | decompose)
       -> parallel retrieval (dense + BM25 + graph, metadata-filtered)
       -> reciprocal rank fusion
       -> rerank (lexical | cross-encoder | llm)
       -> parent-document context expansion
       -> grounded generation with citations
       -> guardrails (output)

Every stage is individually switchable, so the pipeline can be degraded to
plain vector search for comparison - which is exactly what the evaluation
harness does when measuring whether a stage earns its cost.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_studio.core.types import Chunk, Document, Message, RagAnswer, Retrieved
from agentic_studio.guardrails.policy import get_policy
from agentic_studio.observability.logs import get_logger
from agentic_studio.observability.metrics import METRICS
from agentic_studio.observability.tracing import get_tracer
from agentic_studio.rag.chunking import chunk_documents
from agentic_studio.rag.fusion import deduplicate, reciprocal_rank_fusion
from agentic_studio.rag.graph_rag import KnowledgeGraph
from agentic_studio.rag.lexical import BM25Index
from agentic_studio.rag.prompts import NO_CONTEXT_ANSWER, RAG_SYSTEM, build_rag_prompt
from agentic_studio.rag.query_transform import QueryTransformer
from agentic_studio.rag.rerank import build_reranker
from agentic_studio.rag.vector_store import BaseVectorStore, get_vector_store
from agentic_studio.settings import get_settings

logger = get_logger("rag.pipeline")


@dataclass
class RagConfig:
    top_k: int = 8
    fetch_k: int = 30
    hybrid: bool = True
    graph: bool = True
    dense_weight: float = 0.6
    reranker: str = "lexical"
    query_transform: str = "multi-query"
    query_variants: int = 3
    use_parent_context: bool = True
    max_context_chars: int = 12000
    guardrails: bool = True

    @classmethod
    def from_settings(cls) -> RagConfig:
        settings = get_settings()
        return cls(
            top_k=settings.retrieval.top_k,
            fetch_k=settings.retrieval.fetch_k,
            hybrid=settings.retrieval.hybrid_enabled,
            graph=settings.retrieval.graph_rag_enabled,
            dense_weight=settings.retrieval.dense_weight,
            reranker=settings.retrieval.reranker,
            query_transform=settings.retrieval.query_transform,
            query_variants=settings.retrieval.query_variants,
            guardrails=settings.guardrails.enabled,
        )

    def basic(self) -> RagConfig:
        """A deliberately naive configuration, for A/B comparison in evals."""
        return RagConfig(
            top_k=self.top_k,
            fetch_k=self.top_k,
            hybrid=False,
            graph=False,
            reranker="none",
            query_transform="none",
            use_parent_context=False,
            guardrails=self.guardrails,
        )


class RagPipeline:
    def __init__(
        self,
        store: BaseVectorStore | None = None,
        config: RagConfig | None = None,
        router: Any = None,
        index_path: Path | None = None,
    ):
        self.config = config or RagConfig.from_settings()
        # `is not None`, not truthiness: an empty store is falsy but still valid.
        self.store = store if store is not None else get_vector_store(index_path)
        self._router = router
        self.bm25 = BM25Index()
        self.graph = KnowledgeGraph()
        self._reranker = build_reranker(self.config.reranker)
        self._transformer = QueryTransformer(
            self.config.query_transform, self.config.query_variants, router=router
        )
        self._refresh_auxiliary_indexes()

    @property
    def router(self) -> Any:
        if self._router is None:
            from agentic_studio.llm.router import get_router

            self._router = get_router()
        return self._router

    def _refresh_auxiliary_indexes(self) -> None:
        chunks = self.store.all_chunks()
        self.bm25.build(chunks)
        if self.config.graph:
            self.graph.build(chunks)

    # -- ingestion ----------------------------------------------------------

    def ingest_documents(self, documents: list[Document], save: bool = True) -> dict[str, Any]:
        with get_tracer().span("rag.ingest", kind="chain", documents=len(documents)) as span:
            chunks = chunk_documents(documents)
            return self.ingest_chunks(chunks, save=save, span=span)

    def ingest_chunks(self, chunks: list[Chunk], save: bool = True, span: Any = None) -> dict[str, Any]:
        if not chunks:
            return {"chunks_indexed": 0, **self.store.stats()}
        self.store.upsert(chunks)
        if save:
            self.store.save()
        self._refresh_auxiliary_indexes()
        METRICS.incr("rag_chunks_indexed", len(chunks))
        stats = {"chunks_indexed": len(chunks), **self.store.stats()}
        if span is not None:
            span.set(**{k: v for k, v in stats.items() if isinstance(v, (int, float, str))})
        return stats

    def delete(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> int:
        removed = self.store.delete(ids=ids, where=where)
        if removed:
            self.store.save()
            self._refresh_auxiliary_indexes()
        return removed

    # -- retrieval ----------------------------------------------------------

    def retrieve(
        self,
        question: str,
        where: dict[str, Any] | None = None,
        history: list[Message] | None = None,
        top_k: int | None = None,
    ) -> tuple[list[Retrieved], list[str]]:
        top_k = top_k or self.config.top_k

        with get_tracer().span("rag.retrieve", kind="retriever", top_k=top_k) as span:
            queries = self._transformer.transform(question, history)
            rankings: list[list[Retrieved]] = []
            weights: list[float] = []

            for query in queries:
                dense = self.store.search(query, k=self.config.fetch_k, where=where)
                if dense:
                    rankings.append(dense)
                    weights.append(self.config.dense_weight)

                if self.config.hybrid:
                    sparse = self.bm25.search(query, k=self.config.fetch_k, where=where)
                    if sparse:
                        rankings.append(sparse)
                        weights.append(1.0 - self.config.dense_weight)

            if self.config.graph:
                graph_hits = [
                    hit for hit in self.graph.search(question, k=self.config.top_k)
                    if not where or _passes(hit, where)
                ]
                if graph_hits:
                    rankings.append(graph_hits)
                    weights.append(0.4)

            if not rankings:
                span.set(candidates=0)
                return [], queries

            fused = deduplicate(reciprocal_rank_fusion(rankings, weights=weights))
            reranked = self._reranker.rerank(question, fused[: max(self.config.fetch_k, top_k)], top_k=top_k)
            span.set(queries=len(queries), candidates=len(fused), returned=len(reranked))
            METRICS.observe("rag_candidates", len(fused))
            return reranked, queries

    # -- generation ---------------------------------------------------------

    def answer(
        self,
        question: str,
        where: dict[str, Any] | None = None,
        history: list[Message] | None = None,
        top_k: int | None = None,
    ) -> RagAnswer:
        started = time.perf_counter()
        policy = get_policy()
        notes: list[str] = []

        with get_tracer().span("rag.answer", kind="chain") as span:
            if self.config.guardrails:
                verdict = policy.check_input(question)
                question = verdict.raise_if_blocked()
                notes.extend(verdict.notes)

            contexts, queries = self.retrieve(question, where=where, history=history, top_k=top_k)
            if not contexts:
                return RagAnswer(
                    question=question,
                    answer=NO_CONTEXT_ANSWER,
                    queries_used=queries,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    guardrail_notes=notes,
                )

            messages = self._build_messages(question, contexts, history)
            response = self.router.generate(messages)
            answer_text = response.text

            if self.config.guardrails:
                output_verdict = policy.check_output(answer_text)
                answer_text = output_verdict.text
                notes.extend(output_verdict.notes)

            span.set(contexts=len(contexts), tokens=response.usage.total_tokens)
            return RagAnswer(
                question=question,
                answer=answer_text,
                contexts=contexts,
                queries_used=queries,
                usage=response.usage,
                latency_ms=(time.perf_counter() - started) * 1000,
                guardrail_notes=notes,
            )

    def stream_answer(
        self,
        question: str,
        where: dict[str, Any] | None = None,
        history: list[Message] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield retrieval metadata first, then answer tokens.

        The client gets sources immediately and can render them while the model
        is still generating.
        """
        policy = get_policy()
        if self.config.guardrails:
            question = policy.check_input(question).raise_if_blocked()

        contexts, queries = self.retrieve(question, where=where, history=history)
        yield {"type": "sources", "queries": queries, "sources": [c.to_dict() for c in contexts]}

        if not contexts:
            yield {"type": "token", "text": NO_CONTEXT_ANSWER}
            yield {"type": "done"}
            return

        messages = self._build_messages(question, contexts, history)
        collected: list[str] = []
        for chunk in self.router.stream(messages):
            collected.append(chunk)
            yield {"type": "token", "text": chunk}

        final = "".join(collected)
        if self.config.guardrails:
            redacted = policy.check_output(final)
            if redacted.text != final:
                yield {"type": "redacted", "text": redacted.text}
        yield {"type": "done", "chars": len(final)}

    def _build_messages(
        self, question: str, contexts: list[Retrieved], history: list[Message] | None
    ) -> list[Message]:
        policy = get_policy()
        safe_contexts = contexts
        if self.config.guardrails:
            safe_contexts = [
                Retrieved(
                    chunk=Chunk(
                        text=policy.clean_context(item.chunk.text),
                        doc_id=item.chunk.doc_id,
                        metadata=item.chunk.metadata,
                        id=item.chunk.id,
                        parent_text=policy.clean_context(item.chunk.parent_text)
                        if item.chunk.parent_text
                        else None,
                    ),
                    score=item.score,
                    retriever=item.retriever,
                    rank=item.rank,
                )
                for item in contexts
            ]

        messages: list[Message] = [Message.system(RAG_SYSTEM)]
        if history:
            messages.extend(history[-6:])
        prompt = build_rag_prompt(question, safe_contexts, self.config.max_context_chars)
        if not self.config.use_parent_context:
            prompt = build_rag_prompt(
                question,
                [Retrieved(chunk=Chunk(text=c.text, id=c.chunk.id, metadata=c.chunk.metadata),
                           score=c.score, retriever=c.retriever, rank=c.rank)
                 for c in safe_contexts],
                self.config.max_context_chars,
            )
        messages.append(Message.user(prompt))
        return messages

    def stats(self) -> dict[str, Any]:
        return {
            **self.store.stats(),
            "bm25_documents": self.bm25.size,
            "graph_entities": self.graph.node_count,
            "graph_edges": self.graph.edge_count,
            "config": {
                "hybrid": self.config.hybrid,
                "graph": self.config.graph,
                "reranker": self.config.reranker,
                "query_transform": self.config.query_transform,
                "top_k": self.config.top_k,
            },
        }


def _passes(hit: Retrieved, where: dict[str, Any]) -> bool:
    from agentic_studio.rag.vector_store import matches

    return matches(hit.chunk.metadata, where)


_PIPELINE: RagPipeline | None = None


def get_pipeline() -> RagPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = RagPipeline()
    return _PIPELINE


def set_pipeline(pipeline: RagPipeline) -> None:
    global _PIPELINE
    _PIPELINE = pipeline


def reset_pipeline() -> None:
    global _PIPELINE
    _PIPELINE = None
