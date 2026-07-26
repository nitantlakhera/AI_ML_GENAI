"""Ingestion entry points.

Ingestion is incremental and idempotent: chunk ids are derived from the source
path plus chunk index, so re-running over a changed file replaces exactly the
affected chunks instead of duplicating the corpus.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from agentic_studio.core.types import Chunk, Document
from agentic_studio.observability.logs import get_logger
from agentic_studio.rag.chunking import chunk_documents
from agentic_studio.rag.loader import load_documents, load_file, load_texts
from agentic_studio.rag.pipeline import RagPipeline, get_pipeline
from agentic_studio.settings import get_settings

logger = get_logger("rag.ingest")


def stable_chunk_id(chunk: Chunk) -> str:
    """Deterministic id from source + page/row + chunk index + content hash."""
    metadata = chunk.metadata
    parts = [
        str(metadata.get("source", "")),
        str(metadata.get("page", metadata.get("row", metadata.get("record", "")))),
        str(metadata.get("chunk_index", "")),
        hashlib.blake2b(chunk.text.encode("utf-8"), digest_size=8).hexdigest(),
    ]
    digest = hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=12).hexdigest()
    return f"chunk_{digest}"


def prepare_chunks(documents: list[Document]) -> list[Chunk]:
    chunks = chunk_documents(documents)
    for chunk in chunks:
        chunk.id = stable_chunk_id(chunk)
    return chunks


def ingest_directory(
    root: Path | None = None,
    pipeline: RagPipeline | None = None,
    replace_sources: bool = True,
) -> dict[str, Any]:
    """Index every supported file under `root`."""
    root = Path(root or get_settings().paths.data_raw)
    pipeline = pipeline or get_pipeline()

    documents = load_documents(root)
    if not documents:
        return {"chunks_indexed": 0, "documents": 0, "message": f"no documents found in {root}"}

    chunks = prepare_chunks(documents)
    removed = 0
    if replace_sources:
        removed = _drop_stale(pipeline, chunks)

    stats = pipeline.ingest_chunks(chunks)
    stats.update({"documents": len(documents), "stale_chunks_removed": removed, "root": str(root)})
    logger.info("ingested %d chunk(s) from %d document(s)", len(chunks), len(documents))
    return stats


def ingest_file(path: Path, pipeline: RagPipeline | None = None) -> dict[str, Any]:
    pipeline = pipeline or get_pipeline()
    documents = load_file(Path(path))
    if not documents:
        return {"chunks_indexed": 0, "documents": 0, "message": f"nothing extracted from {path}"}
    chunks = prepare_chunks(documents)
    removed = _drop_stale(pipeline, chunks)
    stats = pipeline.ingest_chunks(chunks)
    stats.update({"documents": len(documents), "stale_chunks_removed": removed, "file": str(path)})
    return stats


def ingest_texts(
    texts: list[str], source: str = "inline", pipeline: RagPipeline | None = None
) -> dict[str, Any]:
    pipeline = pipeline or get_pipeline()
    chunks = prepare_chunks(load_texts(texts, source=source))
    stats = pipeline.ingest_chunks(chunks)
    stats.update({"documents": len(texts)})
    return stats


def _drop_stale(pipeline: RagPipeline, incoming: list[Chunk]) -> int:
    """Remove chunks from re-ingested sources that no longer exist in the new run."""
    incoming_ids = {chunk.id for chunk in incoming}
    incoming_sources = {chunk.source for chunk in incoming}
    stale = [
        chunk.id
        for chunk in pipeline.store.all_chunks()
        if chunk.source in incoming_sources and chunk.id not in incoming_ids
    ]
    return pipeline.delete(ids=stale) if stale else 0
