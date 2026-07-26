"""Vector stores with incremental upsert, deletion, and metadata filtering.

`NumpyVectorStore` is the default: exact cosine search, no native dependency,
full persistence. `FaissVectorStore` swaps in FAISS when installed. Both honour
the same interface, so a hosted store (pgvector, Qdrant, Pinecone) only needs
this small surface implemented.
"""

from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from agentic_studio.core.types import Chunk, Retrieved
from agentic_studio.observability.logs import get_logger
from agentic_studio.observability.tracing import get_tracer
from agentic_studio.rag.embeddings import BaseEmbedder, get_embedder
from agentic_studio.settings import get_settings

logger = get_logger("rag.vector_store")


def matches(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """Mongo-flavoured metadata predicate: eq, $in, $nin, $gt(e), $lt(e), $contains."""
    if not where:
        return True
    for key, condition in where.items():
        value = metadata.get(key)
        if isinstance(condition, dict):
            for operator, operand in condition.items():
                if operator == "$in" and value not in operand:
                    return False
                if operator == "$nin" and value in operand:
                    return False
                if operator == "$gt" and not (value is not None and value > operand):
                    return False
                if operator == "$gte" and not (value is not None and value >= operand):
                    return False
                if operator == "$lt" and not (value is not None and value < operand):
                    return False
                if operator == "$lte" and not (value is not None and value <= operand):
                    return False
                if operator == "$contains" and str(operand).lower() not in str(value or "").lower():
                    return False
                if operator == "$ne" and value == operand:
                    return False
        elif value != condition:
            return False
    return True


class BaseVectorStore(ABC):
    """The contract any vector backend must satisfy."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> int:
        ...

    @abstractmethod
    def search(self, query: str, k: int = 8, where: dict[str, Any] | None = None) -> list[Retrieved]:
        ...

    @abstractmethod
    def delete(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> int:
        ...

    @abstractmethod
    def all_chunks(self) -> list[Chunk]:
        ...

    @abstractmethod
    def save(self) -> None:
        ...

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        ...

    def __len__(self) -> int:
        return len(self.all_chunks())


class NumpyVectorStore(BaseVectorStore):
    def __init__(self, path: Path | None = None, embedder: BaseEmbedder | None = None, autoload: bool = True):
        self.path = Path(path or get_settings().paths.index)
        self.embedder = embedder or get_embedder()
        self._vectors: np.ndarray = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._chunks: list[Chunk] = []
        self._index: dict[str, int] = {}
        if autoload:
            self.load()

    # -- writes -------------------------------------------------------------

    def upsert(self, chunks: list[Chunk]) -> int:
        """Insert new chunks and replace existing ones by id. Idempotent re-ingest."""
        if not chunks:
            return 0

        vectors = self.embedder.embed_documents([c.text for c in chunks])
        if vectors.shape[1] != self._vectors.shape[1]:
            if len(self._chunks) == 0:
                self._vectors = np.zeros((0, vectors.shape[1]), dtype=np.float32)
            else:
                raise ValueError(
                    f"embedding dim changed ({self._vectors.shape[1]} -> {vectors.shape[1]}); "
                    "rebuild the index"
                )

        new_rows: list[np.ndarray] = []
        for chunk, vector in zip(chunks, vectors):
            existing = self._index.get(chunk.id)
            if existing is not None:
                self._vectors[existing] = vector
                self._chunks[existing] = chunk
            else:
                self._index[chunk.id] = len(self._chunks) + len(new_rows)
                self._chunks.append(chunk)
                new_rows.append(vector)

        if new_rows:
            self._vectors = np.vstack([self._vectors, np.asarray(new_rows, dtype=np.float32)])
        self._reindex()
        return len(chunks)

    def delete(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> int:
        if not ids and not where:
            return 0
        target = set(ids or [])
        keep = [
            i
            for i, chunk in enumerate(self._chunks)
            if not (chunk.id in target or (where and matches(chunk.metadata, where)))
        ]
        removed = len(self._chunks) - len(keep)
        if removed:
            self._chunks = [self._chunks[i] for i in keep]
            self._vectors = self._vectors[keep] if keep else np.zeros(
                (0, self._vectors.shape[1]), dtype=np.float32
            )
            self._reindex()
        return removed

    def clear(self) -> None:
        self._chunks = []
        self._vectors = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._index = {}

    def _reindex(self) -> None:
        self._index = {chunk.id: i for i, chunk in enumerate(self._chunks)}

    # -- reads --------------------------------------------------------------

    def search(self, query: str, k: int = 8, where: dict[str, Any] | None = None) -> list[Retrieved]:
        if not self._chunks:
            return []
        with get_tracer().span("retriever.dense", kind="retriever", k=k, filtered=bool(where)):
            query_vector = self.embedder.embed_query(query)
            candidates = [i for i, c in enumerate(self._chunks) if matches(c.metadata, where)]
            if not candidates:
                return []
            scores = self._vectors[candidates] @ query_vector
            order = np.argsort(-scores)[:k]
            return [
                Retrieved(
                    chunk=self._chunks[candidates[int(pos)]],
                    score=float(scores[int(pos)]),
                    retriever="dense",
                    rank=rank + 1,
                )
                for rank, pos in enumerate(order)
            ]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def get(self, chunk_id: str) -> Chunk | None:
        position = self._index.get(chunk_id)
        return self._chunks[position] if position is not None else None

    def stats(self) -> dict[str, Any]:
        sources = {c.source for c in self._chunks}
        return {
            "backend": "numpy",
            "chunks": len(self._chunks),
            "sources": len(sources),
            "dim": int(self._vectors.shape[1]) if self._vectors.size else self.embedder.dim,
            "embedder": self.embedder.name,
            "path": str(self.path),
        }

    # -- persistence --------------------------------------------------------

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        np.save(self.path / "vectors.npy", self._vectors)
        (self.path / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self._chunks], ensure_ascii=False), encoding="utf-8"
        )
        (self.path / "manifest.json").write_text(
            json.dumps(
                {"embedder": self.embedder.name, "dim": int(self._vectors.shape[1]) if self._vectors.size
                 else self.embedder.dim, "count": len(self._chunks)},
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("saved %d chunk(s) to %s", len(self._chunks), self.path)

    def load(self) -> bool:
        vectors_path = self.path / "vectors.npy"
        chunks_path = self.path / "chunks.json"
        if not vectors_path.exists() or not chunks_path.exists():
            return False
        try:
            self._vectors = np.load(vectors_path).astype(np.float32)
            self._chunks = [
                Chunk.from_dict(item) for item in json.loads(chunks_path.read_text(encoding="utf-8"))
            ]
            self._reindex()
            logger.info("loaded %d chunk(s) from %s", len(self._chunks), self.path)
            return True
        except Exception as exc:
            logger.warning("failed to load index from %s: %s", self.path, exc)
            self.clear()
            return False

    def destroy(self) -> None:
        self.clear()
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)


class FaissVectorStore(NumpyVectorStore):
    """FAISS-accelerated search; identical semantics, faster at scale.

    Falls back to the numpy path automatically when faiss is not installed.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        # Set before super().__init__ because it calls load(), which rebuilds the index.
        self._faiss = None
        self._faiss_index = None
        try:
            import faiss

            self._faiss = faiss
        except ImportError:
            logger.info("faiss not installed; using exact numpy search")
        super().__init__(*args, **kwargs)

    def _build_faiss(self) -> None:
        if self._faiss is None or self._vectors.size == 0:
            self._faiss_index = None
            return
        index = self._faiss.IndexFlatIP(self._vectors.shape[1])
        index.add(np.ascontiguousarray(self._vectors))
        self._faiss_index = index

    def upsert(self, chunks: list[Chunk]) -> int:
        count = super().upsert(chunks)
        self._build_faiss()
        return count

    def delete(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> int:
        removed = super().delete(ids, where)
        if removed:
            self._build_faiss()
        return removed

    def load(self) -> bool:
        loaded = super().load()
        if loaded:
            self._build_faiss()
        return loaded

    def search(self, query: str, k: int = 8, where: dict[str, Any] | None = None) -> list[Retrieved]:
        # Metadata filters need candidate pre-selection, which flat FAISS cannot
        # express; fall back to exact search in that case.
        if self._faiss_index is None or where:
            return super().search(query, k=k, where=where)

        with get_tracer().span("retriever.dense", kind="retriever", k=k, backend="faiss"):
            query_vector = np.ascontiguousarray(self.embedder.embed_query(query).reshape(1, -1))
            scores, positions = self._faiss_index.search(query_vector, min(k, len(self._chunks)))
            results: list[Retrieved] = []
            for rank, (score, position) in enumerate(zip(scores[0], positions[0])):
                if position < 0:
                    continue
                results.append(
                    Retrieved(chunk=self._chunks[int(position)], score=float(score),
                              retriever="dense", rank=rank + 1)
                )
            return results

    def stats(self) -> dict[str, Any]:
        data = super().stats()
        data["backend"] = "faiss" if self._faiss_index is not None else "numpy"
        return data


def get_vector_store(path: Path | None = None, backend: str = "auto") -> BaseVectorStore:
    """Factory: prefers FAISS when available, always works without it."""
    if backend == "numpy":
        return NumpyVectorStore(path)
    return FaissVectorStore(path)
