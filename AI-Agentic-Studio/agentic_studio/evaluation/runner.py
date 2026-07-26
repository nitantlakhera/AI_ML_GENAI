"""The evaluation runner.

Scores a RAG configuration over a golden set and writes both a JSON report (for
CI thresholds) and a Markdown report (for humans). `compare_configs` runs the
full pipeline against a deliberately naive one so each retrieval stage has to
justify its cost with a number.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentic_studio.core.types import Usage
from agentic_studio.evaluation import metrics as M
from agentic_studio.evaluation.datasets import EvalCase, load_dataset
from agentic_studio.observability.logs import get_logger
from agentic_studio.observability.tracing import get_tracer
from agentic_studio.rag.pipeline import RagConfig, RagPipeline, get_pipeline
from agentic_studio.settings import get_settings

logger = get_logger("evaluation.runner")


@dataclass
class CaseResult:
    case_id: str
    question: str
    answer: str
    scores: dict[str, float]
    latency_ms: float
    contexts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    source_hit: bool | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "answer": self.answer,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "latency_ms": round(self.latency_ms, 2),
            "sources": self.sources,
            "source_hit": self.source_hit,
            "details": self.details,
        }


@dataclass
class EvalReport:
    label: str
    results: list[CaseResult]
    aggregate: dict[str, float]
    usage: Usage
    duration_s: float
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "aggregate": {k: round(v, 4) for k, v in self.aggregate.items()},
            "cases": [result.to_dict() for result in self.results],
            "usage": self.usage.to_dict(),
            "duration_s": round(self.duration_s, 2),
            "config": self.config,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Evaluation report: {self.label}",
            "",
            f"- Cases: **{len(self.results)}**",
            f"- Duration: **{self.duration_s:.1f}s**",
            f"- Tokens: **{self.usage.total_tokens}** (cost ${self.usage.cost_usd:.4f})",
            "",
            "## Aggregate scores",
            "",
            "| Metric | Score |",
            "|--------|-------|",
        ]
        for name, value in self.aggregate.items():
            lines.append(f"| {name} | {value:.3f} |")

        lines += ["", "## Per-case", "", "| Case | Faithful | Relevant | Correct | Citations | Latency |",
                  "|------|----------|----------|---------|-----------|---------|"]
        for result in self.results:
            scores = result.scores
            lines.append(
                f"| {result.case_id} | {scores.get('faithfulness', 0):.2f} | "
                f"{scores.get('answer_relevance', 0):.2f} | "
                f"{scores.get('answer_correctness', 0):.2f} | "
                f"{scores.get('citation_quality', 0):.2f} | {result.latency_ms:.0f}ms |"
            )
        return "\n".join(lines) + "\n"

    def passes(self, thresholds: dict[str, float]) -> tuple[bool, list[str]]:
        """Gate a CI build on minimum scores."""
        failures = [
            f"{name}: {self.aggregate.get(name, 0.0):.3f} < {minimum:.3f}"
            for name, minimum in thresholds.items()
            if self.aggregate.get(name, 0.0) < minimum
        ]
        return not failures, failures


class EvalRunner:
    def __init__(
        self,
        pipeline: RagPipeline | None = None,
        judge: Any = None,
        metrics: tuple[str, ...] = M.ALL_METRICS,
    ):
        self.pipeline = pipeline or get_pipeline()
        self.judge = judge
        self.metrics = metrics

    def run(self, cases: list[EvalCase], label: str = "default") -> EvalReport:
        started = time.perf_counter()
        results: list[CaseResult] = []
        total_usage = Usage()

        with get_tracer().span("eval.run", kind="chain", cases=len(cases), label=label):
            for case in cases:
                result = self.run_case(case)
                results.append(result)

        for result in results:
            total_usage = total_usage + Usage(
                prompt_tokens=result.details.get("prompt_tokens", 0),
                completion_tokens=result.details.get("completion_tokens", 0),
                cost_usd=result.details.get("cost_usd", 0.0),
            )

        return EvalReport(
            label=label,
            results=results,
            aggregate=_aggregate(results),
            usage=total_usage,
            duration_s=time.perf_counter() - started,
            config={
                "hybrid": self.pipeline.config.hybrid,
                "graph": self.pipeline.config.graph,
                "reranker": self.pipeline.config.reranker,
                "query_transform": self.pipeline.config.query_transform,
                "top_k": self.pipeline.config.top_k,
                "judge": self.judge is not None,
            },
        )

    def run_case(self, case: EvalCase) -> CaseResult:
        started = time.perf_counter()
        answer = self.pipeline.answer(case.question, where=case.metadata_filter)
        contexts = [context.text for context in answer.contexts]
        sources = [context.chunk.source for context in answer.contexts]

        scores: dict[str, float] = {}
        details: dict[str, Any] = {
            "prompt_tokens": answer.usage.prompt_tokens,
            "completion_tokens": answer.usage.completion_tokens,
            "cost_usd": answer.usage.cost_usd,
            "queries_used": answer.queries_used,
        }

        if "faithfulness" in self.metrics:
            _record(scores, details, M.faithfulness(answer.answer, contexts, judge=self.judge))
        if "answer_relevance" in self.metrics:
            _record(scores, details, M.answer_relevance(case.question, answer.answer, judge=self.judge))
        if "context_precision" in self.metrics:
            _record(scores, details, M.context_precision(case.question, contexts))
        if "context_recall" in self.metrics and case.reference_answer:
            _record(scores, details, M.context_recall(case.reference_answer, contexts))
        if "answer_correctness" in self.metrics and case.reference_answer:
            _record(
                scores, details,
                M.answer_correctness(answer.answer, case.reference_answer, judge=self.judge),
            )
        if "citation_quality" in self.metrics:
            _record(scores, details, M.citation_quality(answer.answer, len(contexts)))

        source_hit: bool | None = None
        if case.expected_sources:
            source_hit = any(
                any(expected.lower() in source.lower() for source in sources)
                for expected in case.expected_sources
            )
            scores["source_hit"] = 1.0 if source_hit else 0.0

        return CaseResult(
            case_id=case.id or case.question[:40],
            question=case.question,
            answer=answer.answer,
            scores=scores,
            latency_ms=(time.perf_counter() - started) * 1000,
            contexts=contexts,
            sources=sources,
            source_hit=source_hit,
            details=details,
        )


def compare_configs(
    cases: list[EvalCase],
    advanced: RagConfig | None = None,
    judge: Any = None,
) -> dict[str, Any]:
    """Score the advanced pipeline against a naive baseline on the same cases."""
    advanced = advanced or RagConfig.from_settings()
    baseline = advanced.basic()

    advanced_report = EvalRunner(RagPipeline(config=advanced), judge=judge).run(cases, "advanced")
    baseline_report = EvalRunner(RagPipeline(config=baseline), judge=judge).run(cases, "baseline")

    deltas = {
        name: round(advanced_report.aggregate.get(name, 0.0) - value, 4)
        for name, value in baseline_report.aggregate.items()
    }
    return {
        "baseline": baseline_report.to_dict(),
        "advanced": advanced_report.to_dict(),
        "delta": deltas,
    }


def write_report(report: EvalReport, directory: Path | None = None) -> dict[str, Path]:
    directory = Path(directory or get_settings().paths.reports)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = directory / f"eval-{report.label}-{stamp}.json"
    markdown_path = directory / f"eval-{report.label}-{stamp}.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    markdown_path.write_text(report.to_markdown(), encoding="utf-8")
    logger.info("wrote evaluation report to %s", markdown_path)
    return {"json": json_path, "markdown": markdown_path}


def run_from_file(path: Path, label: str = "golden", judge: Any = None) -> EvalReport:
    return EvalRunner(judge=judge).run(load_dataset(path), label=label)


def _record(scores: dict[str, float], details: dict[str, Any], result: M.MetricResult) -> None:
    scores[result.name] = result.score
    if result.detail:
        details[result.name] = result.detail


def _aggregate(results: list[CaseResult]) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for result in results:
        for name, value in result.scores.items():
            totals.setdefault(name, []).append(value)
    aggregate = {name: sum(values) / len(values) for name, values in sorted(totals.items())}
    if aggregate:
        core = [aggregate[k] for k in ("faithfulness", "answer_relevance") if k in aggregate]
        if core:
            aggregate["overall"] = sum(core) / len(core)
    return aggregate
