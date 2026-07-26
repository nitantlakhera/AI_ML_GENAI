"""Evaluation datasets.

A golden set is what turns "the new reranker feels better" into a number. Cases
are JSONL so they diff cleanly in review.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentic_studio.settings import get_settings


@dataclass
class EvalCase:
    question: str
    reference_answer: str = ""
    expected_sources: list[str] = field(default_factory=list)
    metadata_filter: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)
    id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, "", [], {})}


def load_dataset(path: Path) -> list[EvalCase]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"eval dataset not found: {path}")

    cases: list[EvalCase] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        payload = json.loads(line)
        cases.append(
            EvalCase(
                question=payload["question"],
                reference_answer=payload.get("reference_answer", ""),
                expected_sources=list(payload.get("expected_sources", [])),
                metadata_filter=payload.get("metadata_filter"),
                tags=list(payload.get("tags", [])),
                id=payload.get("id") or f"case-{index + 1}",
            )
        )
    return cases


def save_dataset(cases: list[EvalCase], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
    return path


def default_dataset_path() -> Path:
    return get_settings().paths.data_eval / "golden.jsonl"


SAMPLE_CASES = [
    EvalCase(
        id="rrf",
        question="What problem does reciprocal rank fusion solve?",
        reference_answer=(
            "It merges rankings from several retrievers without needing their scores to be "
            "on a comparable scale."
        ),
        tags=["retrieval"],
    ),
    EvalCase(
        id="bm25",
        question="Why combine BM25 with dense retrieval?",
        reference_answer=(
            "BM25 matches exact terms and identifiers that dense embeddings miss, so together "
            "they cover both lexical and semantic matches."
        ),
        tags=["retrieval"],
    ),
    EvalCase(
        id="rerank",
        question="What does a cross-encoder reranker improve?",
        reference_answer="It improves precision at the top of the ranking by scoring the query and document together.",
        tags=["retrieval"],
    ),
    EvalCase(
        id="hitl",
        question="When should an agent pause for human approval?",
        reference_answer=(
            "Before running a tool with side effects, such as writing files, executing code, "
            "or calling a paid or destructive API."
        ),
        tags=["agents"],
    ),
    EvalCase(
        id="absent",
        question="What is the airspeed velocity of an unladen swallow?",
        reference_answer="",
        tags=["out-of-scope", "refusal"],
    ),
]


def write_sample_dataset(path: Path | None = None) -> Path:
    """Create the starter golden set used by `studio eval`."""
    return save_dataset(SAMPLE_CASES, path or default_dataset_path())
