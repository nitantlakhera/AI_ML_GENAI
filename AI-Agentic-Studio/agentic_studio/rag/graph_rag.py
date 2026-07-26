"""Graph RAG - retrieval over an entity co-occurrence graph.

Vector search answers "which chunk looks like this question". A graph answers
"which chunks are connected to the things this question mentions", which is what
multi-hop questions need ("how does X relate to Y" spans two documents that
neither individually resembles the question).

Entities are extracted with capitalisation and frequency heuristics, so this
works without an NER model. Swap `extract_entities` for spaCy or an LLM pass
when quality matters more than startup time.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agentic_studio.core.types import Chunk, Retrieved
from agentic_studio.observability.logs import get_logger
from agentic_studio.observability.tracing import get_tracer
from agentic_studio.rag.lexical import STOPWORDS

logger = get_logger("rag.graph")

_PROPER = re.compile(r"\b([A-Z][a-zA-Z0-9]{2,}(?:\s+[A-Z][a-zA-Z0-9]{2,}){0,3})\b")
_ACRONYM = re.compile(r"\b([A-Z]{2,8})\b")
_TECH = re.compile(r"\b([a-z]+(?:[-_.][a-z0-9]+)+)\b")


def extract_entities(text: str, max_entities: int = 25) -> list[str]:
    """Proper nouns, acronyms, and dotted/hyphenated technical identifiers."""
    found: Counter[str] = Counter()
    for match in _PROPER.findall(text):
        candidate = match.strip()
        if candidate.lower() not in STOPWORDS and len(candidate) > 2:
            found[candidate.lower()] += 2
    for match in _ACRONYM.findall(text):
        found[match.lower()] += 2
    for match in _TECH.findall(text):
        found[match.lower()] += 1
    return [entity for entity, _ in found.most_common(max_entities)]


class KnowledgeGraph:
    """Entities as nodes, co-occurrence as edges, chunks attached to entities."""

    def __init__(self) -> None:
        self.entity_to_chunks: dict[str, set[str]] = defaultdict(set)
        self.edges: dict[str, Counter[str]] = defaultdict(Counter)
        self.chunks: dict[str, Chunk] = {}

    @property
    def node_count(self) -> int:
        return len(self.entity_to_chunks)

    @property
    def edge_count(self) -> int:
        return sum(len(neighbours) for neighbours in self.edges.values()) // 2

    def build(self, chunks: list[Chunk]) -> KnowledgeGraph:
        self.entity_to_chunks.clear()
        self.edges.clear()
        self.chunks.clear()
        self.add(chunks)
        return self

    def add(self, chunks: list[Chunk]) -> KnowledgeGraph:
        for chunk in chunks:
            self.chunks[chunk.id] = chunk
            entities = extract_entities(chunk.text)
            for entity in entities:
                self.entity_to_chunks[entity].add(chunk.id)
            for i, left in enumerate(entities):
                for right in entities[i + 1 :]:
                    self.edges[left][right] += 1
                    self.edges[right][left] += 1
        logger.info("graph: %d entities, %d edges", self.node_count, self.edge_count)
        return self

    def neighbours(self, entity: str, limit: int = 8) -> list[str]:
        return [name for name, _ in self.edges.get(entity, Counter()).most_common(limit)]

    def search(self, query: str, k: int = 8, hops: int = 1) -> list[Retrieved]:
        """Score chunks by how many query entities (and their neighbours) they contain."""
        if not self.entity_to_chunks:
            return []

        seeds = [e for e in extract_entities(query) if e in self.entity_to_chunks]
        if not seeds:
            lowered = query.lower()
            seeds = [entity for entity in self.entity_to_chunks if entity in lowered][:5]
        if not seeds:
            return []

        with get_tracer().span("retriever.graph", kind="retriever", seeds=len(seeds), hops=hops):
            weights: dict[str, float] = {entity: 1.0 for entity in seeds}
            frontier = list(seeds)
            for hop in range(1, hops + 1):
                decay = 1.0 / (2**hop)
                next_frontier: list[str] = []
                for entity in frontier:
                    for neighbour in self.neighbours(entity):
                        if neighbour not in weights:
                            weights[neighbour] = decay
                            next_frontier.append(neighbour)
                frontier = next_frontier

            scores: dict[str, float] = defaultdict(float)
            for entity, weight in weights.items():
                for chunk_id in self.entity_to_chunks.get(entity, ()):  # type: ignore[arg-type]
                    scores[chunk_id] += weight

            ordered = sorted(scores.items(), key=lambda pair: -pair[1])[:k]
            return [
                Retrieved(chunk=self.chunks[chunk_id], score=float(score), retriever="graph",
                          rank=rank + 1)
                for rank, (chunk_id, score) in enumerate(ordered)
                if chunk_id in self.chunks
            ]

    def related(self, entity: str, hops: int = 1) -> dict[str, Any]:
        """Inspect the neighbourhood of one entity; used by the graph explorer tool."""
        entity = entity.lower().strip()
        if entity not in self.entity_to_chunks:
            return {"entity": entity, "found": False, "neighbours": [], "chunks": []}
        neighbours = self.neighbours(entity, limit=12)
        if hops > 1:
            expanded = list(neighbours)
            for neighbour in neighbours:
                expanded.extend(self.neighbours(neighbour, limit=4))
            neighbours = list(dict.fromkeys(expanded))
        return {
            "entity": entity,
            "found": True,
            "neighbours": neighbours,
            "chunks": [
                {"id": chunk_id, "source": self.chunks[chunk_id].source}
                for chunk_id in list(self.entity_to_chunks[entity])[:10]
                if chunk_id in self.chunks
            ],
        }

    # -- persistence --------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "entity_to_chunks": {k: sorted(v) for k, v in self.entity_to_chunks.items()},
            "edges": {k: dict(v) for k, v in self.edges.items()},
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def load(self, path: Path, chunks: list[Chunk]) -> bool:
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("failed to load graph: %s", exc)
            return False
        self.entity_to_chunks = defaultdict(set, {k: set(v) for k, v in payload["entity_to_chunks"].items()})
        self.edges = defaultdict(Counter, {k: Counter(v) for k, v in payload["edges"].items()})
        self.chunks = {chunk.id: chunk for chunk in chunks}
        return True
