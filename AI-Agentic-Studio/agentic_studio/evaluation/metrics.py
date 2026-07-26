"""RAG and answer quality metrics.

Every metric has a deterministic lexical/embedding implementation so evaluation
runs offline, in CI, and without spending tokens. Pass `judge=...` to upgrade
the semantic ones to LLM-as-judge scoring where nuance matters.

Metric map:
    faithfulness       - is every claim in the answer supported by the context?
    answer_relevance   - does the answer address the question that was asked?
    context_precision  - what fraction of retrieved chunks were actually useful?
    context_recall     - did retrieval find what the reference answer needs?
    answer_correctness - how close is the answer to the reference?
    citation_quality   - are claims cited, and do the citations point somewhere real?
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agentic_studio.rag.lexical import tokenize

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class MetricResult:
    name: str
    score: float
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "score": round(self.score, 4), "detail": self.detail}


def sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE.split(text or "") if len(part.strip()) > 3]


def token_overlap(left: str, right: str) -> float:
    """Proportion of `left`'s terms that also appear in `right`."""
    left_terms = set(tokenize(left))
    if not left_terms:
        return 0.0
    right_terms = set(tokenize(right))
    return len(left_terms & right_terms) / len(left_terms)


def token_f1(prediction: str, reference: str) -> float:
    predicted = set(tokenize(prediction))
    expected = set(tokenize(reference))
    if not predicted or not expected:
        return 0.0
    shared = len(predicted & expected)
    if shared == 0:
        return 0.0
    precision = shared / len(predicted)
    recall = shared / len(expected)
    return 2 * precision * recall / (precision + recall)


def embedding_similarity(left: str, right: str) -> float:
    if not left.strip() or not right.strip():
        return 0.0
    from agentic_studio.rag.embeddings import get_embedder

    vectors = get_embedder().embed_documents([left, right])
    return float(np.clip(np.dot(vectors[0], vectors[1]), 0.0, 1.0))


# -- metrics ----------------------------------------------------------------


def faithfulness(
    answer: str, contexts: list[str], threshold: float = 0.55, judge: Any = None
) -> MetricResult:
    """Fraction of answer claims supported by at least one retrieved context."""
    claims = sentences(answer)
    if not claims:
        return MetricResult("faithfulness", 0.0, {"claims": 0})
    if not contexts:
        return MetricResult("faithfulness", 0.0, {"claims": len(claims), "contexts": 0})

    if judge is not None:
        return judge.faithfulness(answer, contexts)

    joined = "\n".join(contexts)
    supported: list[str] = []
    unsupported: list[str] = []
    for claim in claims:
        best = max(token_overlap(claim, context) for context in contexts)
        if best < threshold:
            best = max(best, token_overlap(claim, joined))
        (supported if best >= threshold else unsupported).append(claim)

    return MetricResult(
        "faithfulness",
        len(supported) / len(claims),
        {"claims": len(claims), "unsupported": unsupported[:5]},
    )


def answer_relevance(question: str, answer: str, judge: Any = None) -> MetricResult:
    """How much of the question the answer actually engages with."""
    if not answer.strip():
        return MetricResult("answer_relevance", 0.0, {"reason": "empty answer"})
    if judge is not None:
        return judge.answer_relevance(question, answer)

    lexical = token_overlap(question, answer)
    semantic = embedding_similarity(question, answer)
    refusal = _is_refusal(answer)
    score = 0.4 * lexical + 0.6 * semantic
    if refusal:
        score *= 0.5
    return MetricResult(
        "answer_relevance",
        float(np.clip(score, 0.0, 1.0)),
        {"lexical": round(lexical, 4), "semantic": round(semantic, 4), "refusal": refusal},
    )


def context_precision(question: str, contexts: list[str], threshold: float = 0.2) -> MetricResult:
    """Fraction of retrieved contexts that are relevant to the question.

    Low precision means the LLM is paying for tokens it cannot use, and is more
    likely to be distracted into a wrong answer.
    """
    if not contexts:
        return MetricResult("context_precision", 0.0, {"contexts": 0})
    relevant = [context for context in contexts if token_overlap(question, context) >= threshold]
    return MetricResult(
        "context_precision",
        len(relevant) / len(contexts),
        {"contexts": len(contexts), "relevant": len(relevant)},
    )


def context_recall(reference: str, contexts: list[str], threshold: float = 0.5) -> MetricResult:
    """Fraction of the reference answer's claims that retrieval actually found."""
    claims = sentences(reference)
    if not claims:
        return MetricResult("context_recall", 0.0, {"claims": 0})
    if not contexts:
        return MetricResult("context_recall", 0.0, {"claims": len(claims), "contexts": 0})

    joined = "\n".join(contexts)
    covered = [claim for claim in claims if token_overlap(claim, joined) >= threshold]
    return MetricResult(
        "context_recall",
        len(covered) / len(claims),
        {"claims": len(claims), "covered": len(covered)},
    )


def answer_correctness(answer: str, reference: str, judge: Any = None) -> MetricResult:
    """Agreement between the produced answer and the reference answer."""
    if not reference.strip():
        return MetricResult("answer_correctness", 0.0, {"reason": "no reference"})
    if judge is not None:
        return judge.answer_correctness(answer, reference)

    lexical = token_f1(answer, reference)
    semantic = embedding_similarity(answer, reference)
    return MetricResult(
        "answer_correctness",
        float(np.clip(0.5 * lexical + 0.5 * semantic, 0.0, 1.0)),
        {"token_f1": round(lexical, 4), "semantic": round(semantic, 4)},
    )


def citation_quality(answer: str, context_count: int) -> MetricResult:
    """Are claims cited, and do the citation numbers point at real contexts?"""
    claims = sentences(answer)
    if not claims:
        return MetricResult("citation_quality", 0.0, {"claims": 0})

    cited = [claim for claim in claims if _CITATION.search(claim)]
    referenced = {int(index) for index in _CITATION.findall(answer)}
    invalid = sorted(index for index in referenced if index < 1 or index > context_count)

    coverage = len(cited) / len(claims)
    validity = 1.0 if not referenced else 1.0 - len(invalid) / len(referenced)
    return MetricResult(
        "citation_quality",
        float(np.clip(0.6 * coverage + 0.4 * validity, 0.0, 1.0)),
        {
            "claims": len(claims),
            "cited_claims": len(cited),
            "citations": sorted(referenced),
            "invalid_citations": invalid,
        },
    )


def _is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    markers = (
        "i could not find", "does not contain", "not enough information",
        "cannot answer", "no relevant", "i don't know", "i do not know",
    )
    return any(marker in lowered for marker in markers)


ALL_METRICS = (
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "answer_correctness",
    "citation_quality",
)
