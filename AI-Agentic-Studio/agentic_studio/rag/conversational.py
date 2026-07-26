"""Conversational RAG.

Turns the stateless pipeline into a multi-turn assistant: history is persisted,
compacted when it grows, and used to rewrite follow-up questions into standalone
retrieval queries ("what about the second one?" is unanswerable on its own).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from agentic_studio.core.types import Message, RagAnswer
from agentic_studio.memory.summarizing import SummarizingMemory
from agentic_studio.observability.tracing import get_tracer
from agentic_studio.rag.pipeline import RagPipeline, get_pipeline
from agentic_studio.rag.query_transform import QueryTransformer


class ConversationalRag:
    def __init__(
        self,
        pipeline: RagPipeline | None = None,
        memory: SummarizingMemory | None = None,
        rewrite_followups: bool = True,
    ):
        self.pipeline = pipeline or get_pipeline()
        self.memory = memory or SummarizingMemory()
        self.rewrite_followups = rewrite_followups
        self._rewriter = QueryTransformer(strategy="rewrite", router=self.pipeline._router)

    def ask(
        self,
        thread_id: str,
        question: str,
        where: dict[str, Any] | None = None,
    ) -> RagAnswer:
        with get_tracer().span("rag.conversational", kind="chain", thread=thread_id):
            history = self.memory.load(thread_id)
            search_question = self._standalone(question, history)

            answer = self.pipeline.answer(search_question, where=where, history=history)
            answer.question = question
            if search_question != question:
                answer.queries_used = [search_question, *answer.queries_used]

            self.memory.append(thread_id, Message.user(question))
            self.memory.append(thread_id, Message.assistant(answer.answer))
            return answer

    def stream(
        self,
        thread_id: str,
        question: str,
        where: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        history = self.memory.load(thread_id)
        search_question = self._standalone(question, history)

        collected: list[str] = []
        for event in self.pipeline.stream_answer(search_question, where=where, history=history):
            if event.get("type") == "token":
                collected.append(event["text"])
            yield event

        self.memory.append(thread_id, Message.user(question))
        self.memory.append(thread_id, Message.assistant("".join(collected)))

    def history(self, thread_id: str) -> list[Message]:
        return self.memory.store.history(thread_id)

    def reset(self, thread_id: str) -> bool:
        return self.memory.clear(thread_id)

    def _standalone(self, question: str, history: list[Message]) -> str:
        if not self.rewrite_followups or not history:
            return question
        try:
            return self._rewriter.rewrite(question, history)
        except Exception:
            return question
