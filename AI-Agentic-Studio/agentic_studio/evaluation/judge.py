"""LLM-as-judge scoring.

Used to upgrade the semantic metrics when a real model is configured. Every
method degrades to the deterministic metric on any failure, so enabling the
judge can improve scores' nuance but can never break a run.
"""

from __future__ import annotations

from typing import Any

from agentic_studio.evaluation.metrics import MetricResult
from agentic_studio.observability.logs import get_logger

logger = get_logger("evaluation.judge")

_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
}


class LLMJudge:
    def __init__(self, router: Any = None, scale: float = 5.0):
        self._router = router
        self.scale = scale

    @property
    def router(self) -> Any:
        if self._router is None:
            from agentic_studio.llm.router import get_router

            self._router = get_router()
        return self._router

    def _grade(self, name: str, prompt: str, fallback: Any) -> MetricResult:
        from agentic_studio.llm.structured import generate_structured

        try:
            parsed = generate_structured(prompt, _SCORE_SCHEMA, router=self.router, retries=1)
            raw = float(parsed.get("score", 0.0))
        except Exception as exc:
            logger.warning("judge failed for %s (%s); using deterministic metric", name, exc)
            return fallback()

        if raw <= 0:
            # A zero usually means the model returned a placeholder rather than a
            # judgement; the deterministic metric is more informative there.
            return fallback()

        score = max(0.0, min(1.0, raw / self.scale if raw > 1.0 else raw))
        return MetricResult(name, score, {"judge": True, "raw": raw,
                                          "reason": parsed.get("reason", "")[:300]})

    def faithfulness(self, answer: str, contexts: list[str]) -> MetricResult:
        from agentic_studio.evaluation import metrics as deterministic

        context_block = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1))
        prompt = (
            "Score how well the answer is supported by the context, from 0 to 5.\n"
            "5 = every claim is directly supported. 0 = the answer contradicts or invents facts.\n\n"
            f"Context:\n{context_block}\n\nAnswer:\n{answer}"
        )
        return self._grade(
            "faithfulness",
            prompt,
            lambda: deterministic.faithfulness(answer, contexts),
        )

    def answer_relevance(self, question: str, answer: str) -> MetricResult:
        from agentic_studio.evaluation import metrics as deterministic

        prompt = (
            "Score how directly the answer addresses the question, from 0 to 5.\n"
            "Ignore whether it is factually correct; judge relevance only.\n\n"
            f"Question: {question}\n\nAnswer: {answer}"
        )
        return self._grade(
            "answer_relevance",
            prompt,
            lambda: deterministic.answer_relevance(question, answer),
        )

    def answer_correctness(self, answer: str, reference: str) -> MetricResult:
        from agentic_studio.evaluation import metrics as deterministic

        prompt = (
            "Score how well the answer matches the reference answer, from 0 to 5.\n"
            "Different wording with the same meaning scores high.\n\n"
            f"Reference: {reference}\n\nAnswer: {answer}"
        )
        return self._grade(
            "answer_correctness",
            prompt,
            lambda: deterministic.answer_correctness(answer, reference),
        )

    def compare(self, question: str, answer_a: str, answer_b: str) -> dict[str, Any]:
        """Pairwise preference, for A/B testing two pipeline configurations."""
        from agentic_studio.llm.structured import generate_structured

        schema = {
            "type": "object",
            "properties": {
                "winner": {"type": "string", "enum": ["A", "B", "tie"]},
                "reason": {"type": "string"},
            },
            "required": ["winner"],
        }
        prompt = (
            f"Question: {question}\n\nAnswer A:\n{answer_a}\n\nAnswer B:\n{answer_b}\n\n"
            "Which answer is more accurate, better grounded, and more useful?"
        )
        try:
            return generate_structured(prompt, schema, router=self.router, retries=1)
        except Exception as exc:
            return {"winner": "tie", "reason": f"judge unavailable: {exc}"}
