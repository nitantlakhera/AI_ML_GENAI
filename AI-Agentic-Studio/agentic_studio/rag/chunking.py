"""Chunking strategies.

Three strategies, all producing parent-document context so the retriever can
match on a small precise chunk while the LLM reads a wider window:

* `recursive` - split on paragraph/sentence/word boundaries (safe default)
* `semantic`  - split where consecutive sentence embeddings diverge
* `markdown`  - split on heading structure, keeping the heading trail
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from agentic_studio.core.types import Chunk, Document
from agentic_studio.observability.logs import get_logger
from agentic_studio.settings import get_settings

logger = get_logger("rag.chunking")

_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def chunk_documents(
    documents: list[Document],
    strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    parent_chunk_size: int | None = None,
) -> list[Chunk]:
    settings = get_settings().retrieval
    strategy = (strategy or settings.chunk_strategy).lower()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    parent_chunk_size = parent_chunk_size or settings.parent_chunk_size

    chunks: list[Chunk] = []
    for document in documents:
        if strategy == "semantic":
            pieces = semantic_split(document.text, chunk_size)
        elif strategy == "markdown":
            pieces = markdown_split(document.text, chunk_size, chunk_overlap)
        else:
            pieces = recursive_split(document.text, chunk_size, chunk_overlap)

        parents = recursive_split(document.text, parent_chunk_size, 0)
        for index, (text, local_meta) in enumerate(pieces):
            if not text.strip():
                continue
            chunks.append(
                Chunk(
                    text=text.strip(),
                    doc_id=document.id,
                    metadata={
                        **document.metadata,
                        **local_meta,
                        "chunk_index": index,
                        "chunk_strategy": strategy,
                    },
                    parent_text=_find_parent(text, parents),
                )
            )
    logger.info("produced %d chunk(s) with strategy=%s", len(chunks), strategy)
    return chunks


def recursive_split(
    text: str, chunk_size: int, chunk_overlap: int, separators: list[str] | None = None
) -> list[tuple[str, dict[str, Any]]]:
    """Split on the largest separator that keeps pieces under `chunk_size`."""
    pieces = _recursive(text, chunk_size, separators or _SEPARATORS)
    merged = _merge(pieces, chunk_size, chunk_overlap)
    return [(piece, {}) for piece in merged]


def _recursive(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []
    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator, *rest = separators
    parts = text.split(separator)
    if len(parts) == 1:
        return _recursive(text, chunk_size, rest)

    output: list[str] = []
    for part in parts:
        if len(part) <= chunk_size:
            if part.strip():
                output.append(part)
        else:
            output.extend(_recursive(part, chunk_size, rest))
    return output


def _merge(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack small pieces up to chunk_size, carrying an overlap tail."""
    merged: list[str] = []
    buffer = ""
    for piece in pieces:
        candidate = f"{buffer} {piece}".strip() if buffer else piece
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue
        if buffer:
            merged.append(buffer)
            tail = buffer[-overlap:] if overlap > 0 else ""
            buffer = f"{tail} {piece}".strip() if tail else piece
        else:
            merged.append(piece[:chunk_size])
            buffer = piece[chunk_size:]
    if buffer.strip():
        merged.append(buffer.strip())
    return merged


def semantic_split(text: str, max_chars: int, threshold: float = 0.55) -> list[tuple[str, dict[str, Any]]]:
    """Break where the topic shifts, measured by sentence-embedding similarity."""
    sentences = [s.strip() for s in _SENTENCE.split(text) if s.strip()]
    if len(sentences) < 3:
        return recursive_split(text, max_chars, 0)

    from agentic_studio.rag.embeddings import get_embedder

    vectors = get_embedder().embed_documents(sentences)
    similarities = [float(np.dot(vectors[i], vectors[i + 1])) for i in range(len(sentences) - 1)]

    groups: list[list[str]] = [[sentences[0]]]
    for index, similarity in enumerate(similarities):
        current = groups[-1]
        too_long = len(" ".join(current)) + len(sentences[index + 1]) > max_chars
        if similarity < threshold or too_long:
            groups.append([sentences[index + 1]])
        else:
            current.append(sentences[index + 1])

    return [(" ".join(group), {"split": "semantic"}) for group in groups]


def markdown_split(text: str, chunk_size: int, overlap: int) -> list[tuple[str, dict[str, Any]]]:
    """Split by heading, keeping the heading trail as metadata for citations."""
    sections: list[tuple[str, list[str]]] = []
    trail: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer and "".join(buffer).strip():
            sections.append(("\n".join(buffer).strip(), list(trail)))

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            flush()
            buffer = []
            level = len(match.group(1))
            trail = trail[: level - 1]
            trail.append(match.group(2).strip())
            buffer.append(line)
        else:
            buffer.append(line)
    flush()

    output: list[tuple[str, dict[str, Any]]] = []
    for body, heading_trail in sections:
        meta = {"heading": " > ".join(heading_trail), "split": "markdown"}
        if len(body) <= chunk_size:
            output.append((body, meta))
        else:
            for piece, _ in recursive_split(body, chunk_size, overlap):
                output.append((piece, meta))
    return output


def _find_parent(chunk_text: str, parents: list[tuple[str, dict[str, Any]]]) -> str | None:
    """Attach the wider window that contains this chunk, when there is one."""
    probe = chunk_text.strip()[:60]
    if not probe:
        return None
    for parent_text, _ in parents:
        if probe in parent_text:
            return parent_text if len(parent_text) > len(chunk_text) else None
    return None
