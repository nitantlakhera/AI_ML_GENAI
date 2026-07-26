"""Result fusion for hybrid retrieval.

Reciprocal Rank Fusion is the default because it combines rankings without
needing the retrievers' scores to be on a comparable scale - a dense cosine of
0.82 and a BM25 score of 14.3 are not otherwise addable.
"""

from __future__ import annotations

from collections.abc import Sequence

from agentic_studio.core.types import Retrieved


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Retrieved]],
    k: int = 60,
    weights: Sequence[float] | None = None,
    top_k: int | None = None,
) -> list[Retrieved]:
    """RRF: score(d) = sum over lists of weight / (k + rank(d))."""
    weights = list(weights) if weights else [1.0] * len(rankings)
    if len(weights) < len(rankings):
        weights += [1.0] * (len(rankings) - len(weights))

    scores: dict[str, float] = {}
    best: dict[str, Retrieved] = {}
    contributors: dict[str, set[str]] = {}

    for list_index, ranking in enumerate(rankings):
        weight = weights[list_index]
        for rank, item in enumerate(ranking, start=1):
            key = item.chunk.id
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            contributors.setdefault(key, set()).add(item.retriever)
            if key not in best or rank < best[key].rank:
                best[key] = item

    ordered = sorted(scores.items(), key=lambda pair: -pair[1])
    if top_k is not None:
        ordered = ordered[:top_k]

    fused: list[Retrieved] = []
    for rank, (key, score) in enumerate(ordered, start=1):
        source = best[key]
        fused.append(
            Retrieved(
                chunk=source.chunk,
                score=score,
                retriever="+".join(sorted(contributors[key])),
                rank=rank,
            )
        )
    return fused


def weighted_score_fusion(
    rankings: Sequence[Sequence[Retrieved]],
    weights: Sequence[float] | None = None,
    top_k: int | None = None,
) -> list[Retrieved]:
    """Min-max normalise each list, then take a weighted sum.

    Useful when the absolute scores carry signal you do not want to discard.
    """
    weights = list(weights) if weights else [1.0] * len(rankings)
    scores: dict[str, float] = {}
    best: dict[str, Retrieved] = {}
    contributors: dict[str, set[str]] = {}

    for list_index, ranking in enumerate(rankings):
        if not ranking:
            continue
        values = [item.score for item in ranking]
        low, high = min(values), max(values)
        span = (high - low) or 1.0
        weight = weights[list_index] if list_index < len(weights) else 1.0
        for item in ranking:
            key = item.chunk.id
            normalized = (item.score - low) / span
            scores[key] = scores.get(key, 0.0) + weight * normalized
            contributors.setdefault(key, set()).add(item.retriever)
            best.setdefault(key, item)

    ordered = sorted(scores.items(), key=lambda pair: -pair[1])
    if top_k is not None:
        ordered = ordered[:top_k]

    return [
        Retrieved(
            chunk=best[key].chunk,
            score=score,
            retriever="+".join(sorted(contributors[key])),
            rank=rank,
        )
        for rank, (key, score) in enumerate(ordered, start=1)
    ]


def deduplicate(results: Sequence[Retrieved]) -> list[Retrieved]:
    """Drop repeated chunks, keeping the highest-ranked occurrence."""
    seen: set[str] = set()
    unique: list[Retrieved] = []
    for item in results:
        if item.chunk.id in seen:
            continue
        seen.add(item.chunk.id)
        unique.append(item)
    for rank, item in enumerate(unique, start=1):
        item.rank = rank
    return unique
