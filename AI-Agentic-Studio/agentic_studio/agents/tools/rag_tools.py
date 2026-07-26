"""RAG exposed as agent tools.

This is the piece that connects the two halves of the studio: the agent can
consult the document corpus, explore the knowledge graph, and inspect the index
the same way it uses any other tool.
"""

from __future__ import annotations

from typing import Any

from agentic_studio.agents.tools.registry import tool


def _pipeline():
    from agentic_studio.rag.pipeline import get_pipeline

    return get_pipeline()


@tool(name="rag_search", tags=("research", "rag"))
def rag_search(query: str, top_k: int = 5, source_contains: str = "") -> dict[str, Any]:
    """Search the ingested document corpus and return the most relevant passages.

    Args:
        query: What to look for in the indexed documents.
        top_k: How many passages to return.
        source_contains: Optional filter; only match sources containing this text.
    """
    where = {"source": {"$contains": source_contains}} if source_contains else None
    hits, queries = _pipeline().retrieve(query, where=where, top_k=max(1, min(int(top_k), 15)))
    return {
        "query": query,
        "queries_searched": queries,
        "count": len(hits),
        "passages": [
            {
                "rank": hit.rank,
                "source": hit.chunk.source,
                "title": hit.chunk.metadata.get("title", ""),
                "page": hit.chunk.metadata.get("page"),
                "score": round(hit.score, 4),
                "text": hit.text[:1200],
            }
            for hit in hits
        ],
    }


@tool(name="rag_answer", tags=("research", "rag"))
def rag_answer(question: str) -> dict[str, Any]:
    """Answer a question strictly from the ingested documents, with citations.

    Args:
        question: The question to answer from the corpus.
    """
    result = _pipeline().answer(question)
    return {
        "answer": result.answer,
        "citations": [
            {"n": index, "source": context.chunk.source, "page": context.chunk.metadata.get("page")}
            for index, context in enumerate(result.contexts, start=1)
        ],
    }


@tool(name="graph_explore", tags=("research", "rag"))
def graph_explore(entity: str, hops: int = 1) -> dict[str, Any]:
    """Explore entities connected to a term in the knowledge graph.

    Args:
        entity: The entity or term to look up.
        hops: How many hops to expand, 1 or 2.
    """
    return _pipeline().graph.related(entity, hops=max(1, min(int(hops), 2)))


@tool(name="corpus_stats", tags=("rag",))
def corpus_stats() -> dict[str, Any]:
    """Report what is currently indexed: chunk count, sources, and retrieval settings."""
    return _pipeline().stats()


@tool(name="list_sources", tags=("rag",))
def list_sources(limit: int = 50) -> dict[str, Any]:
    """List the distinct document sources present in the index.

    Args:
        limit: Maximum number of sources to return.
    """
    counts: dict[str, int] = {}
    for chunk in _pipeline().store.all_chunks():
        counts[chunk.source] = counts.get(chunk.source, 0) + 1
    ordered = sorted(counts.items(), key=lambda pair: -pair[1])[: max(1, int(limit))]
    return {
        "total_sources": len(counts),
        "sources": [{"source": source, "chunks": count} for source, count in ordered],
    }
