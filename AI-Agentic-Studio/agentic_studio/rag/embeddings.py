"""Embedding backends.

`hashing` is the default: deterministic, dependency-free, no model download, and
good enough for lexical-overlap similarity - which keeps tests and demos fast
and offline. Switch to `sentence-transformers` for real semantic quality.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np

from agentic_studio.observability.logs import get_logger
from agentic_studio.settings import get_settings

logger = get_logger("rag.embeddings")

_TOKEN = re.compile(r"[a-z0-9']+")


class BaseEmbedder(ABC):
    name: str = "base"

    def __init__(self, dim: int):
        self.dim = dim

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        ...

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    @staticmethod
    def normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class HashingEmbedder(BaseEmbedder):
    """Signed feature hashing over word unigrams, bigrams, and char 4-grams.

    Deterministic across processes and machines, so cached vectors and committed
    indexes stay valid. Similarity approximates weighted lexical overlap.
    """

    name = "hashing"

    def __init__(self, dim: int = 384):
        super().__init__(dim)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, weight in self._features(text):
                bucket, sign = self._bucket(feature)
                matrix[row, bucket] += sign * weight
        return self.normalize(matrix)

    def _features(self, text: str) -> list[tuple[str, float]]:
        tokens = _TOKEN.findall(text.lower())
        features: list[tuple[str, float]] = [(token, 1.0) for token in tokens]
        features.extend((f"{a}_{b}", 0.6) for a, b in zip(tokens, tokens[1:]))
        compact = "".join(tokens)
        features.extend((compact[i : i + 4], 0.25) for i in range(0, max(0, len(compact) - 3), 2))
        return features

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0


class SentenceTransformerEmbedder(BaseEmbedder):
    """Real semantic embeddings via sentence-transformers (optional extra)."""

    name = "sentence-transformers"

    def __init__(self, model_name: str, dim: int = 384):
        super().__init__(dim)
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                               show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


def build_embedder(backend: str | None = None, model: str | None = None, dim: int | None = None) -> BaseEmbedder:
    settings = get_settings().retrieval
    backend = (backend or settings.embedding_backend).strip().lower()
    dim = dim or settings.embedding_dim

    if backend in {"sentence-transformers", "st", "huggingface"}:
        try:
            return SentenceTransformerEmbedder(model or settings.embedding_model, dim=dim)
        except ImportError:
            logger.warning("sentence-transformers not installed; falling back to hashing embedder")
            return HashingEmbedder(dim=dim)
    if backend != "hashing":
        logger.warning("unknown embedding backend %r; using hashing", backend)
    return HashingEmbedder(dim=dim)


@lru_cache(maxsize=4)
def get_embedder() -> BaseEmbedder:
    return build_embedder()


def reset_embedder() -> None:
    get_embedder.cache_clear()
