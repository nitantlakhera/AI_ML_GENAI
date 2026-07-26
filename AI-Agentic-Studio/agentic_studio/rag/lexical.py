"""BM25 keyword retrieval.

Dense retrieval misses exact identifiers, error codes, product names, and rare
terms. BM25 catches precisely those, which is why hybrid search beats either
retriever alone. Implemented here so there is no extra dependency.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from agentic_studio.core.types import Chunk, Retrieved
from agentic_studio.observability.tracing import get_tracer
from agentic_studio.rag.vector_store import matches

_TOKEN = re.compile(r"[a-z0-9][a-z0-9'_\-.]*")

STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "had", "has",
    "have", "he", "her", "his", "i", "if", "in", "into", "is", "it", "its", "of", "on", "or",
    "our", "she", "so", "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "to", "was", "we", "were", "what", "when", "which", "who", "will", "with", "you", "your",
}


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    if drop_stopwords:
        return [t for t in tokens if t not in STOPWORDS]
    return tokens


class BM25Index:
    """Okapi BM25 over chunk text, rebuildable incrementally."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._term_freqs: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._doc_freq: Counter[str] = Counter()
        self._avg_length = 0.0

    @property
    def size(self) -> int:
        return len(self._chunks)

    def build(self, chunks: list[Chunk]) -> BM25Index:
        self._chunks = []
        self._term_freqs = []
        self._lengths = []
        self._doc_freq = Counter()
        self.add(chunks)
        return self

    def add(self, chunks: list[Chunk]) -> BM25Index:
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            frequencies = Counter(tokens)
            self._chunks.append(chunk)
            self._term_freqs.append(frequencies)
            self._lengths.append(len(tokens))
            for term in frequencies:
                self._doc_freq[term] += 1
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        return self

    def _idf(self, term: str) -> float:
        total = len(self._chunks)
        frequency = self._doc_freq.get(term, 0)
        if frequency == 0:
            return 0.0
        return math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))

    def search(self, query: str, k: int = 8, where: dict[str, Any] | None = None) -> list[Retrieved]:
        if not self._chunks:
            return []
        terms = tokenize(query)
        if not terms:
            return []

        with get_tracer().span("retriever.bm25", kind="retriever", k=k):
            scored: list[tuple[float, int]] = []
            for position, frequencies in enumerate(self._term_freqs):
                if where and not matches(self._chunks[position].metadata, where):
                    continue
                length = self._lengths[position] or 1
                score = 0.0
                for term in terms:
                    frequency = frequencies.get(term, 0)
                    if frequency == 0:
                        continue
                    denominator = frequency + self.k1 * (
                        1 - self.b + self.b * length / (self._avg_length or 1)
                    )
                    score += self._idf(term) * frequency * (self.k1 + 1) / denominator
                if score > 0:
                    scored.append((score, position))

            scored.sort(key=lambda item: -item[0])
            return [
                Retrieved(chunk=self._chunks[position], score=float(score), retriever="bm25",
                          rank=rank + 1)
                for rank, (score, position) in enumerate(scored[:k])
            ]
