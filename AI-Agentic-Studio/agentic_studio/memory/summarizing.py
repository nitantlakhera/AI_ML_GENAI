"""Token-budgeted memory.

Long conversations eventually exceed the context window. This keeps the last N
turns verbatim and folds everything older into a running summary, so a thread
can run indefinitely at bounded cost.
"""

from __future__ import annotations

from typing import Any

from agentic_studio.core.types import Message
from agentic_studio.memory.store import ConversationStore, get_store
from agentic_studio.observability.logs import get_logger
from agentic_studio.observability.metrics import estimate_tokens
from agentic_studio.observability.tracing import get_tracer

logger = get_logger("memory.summarizing")

SUMMARY_PROMPT = (
    "Update the running summary of this conversation. Keep facts, decisions, "
    "names, numbers, and open questions. Drop small talk. Write at most 200 words.\n\n"
    "Existing summary:\n{summary}\n\nNew turns:\n{turns}\n\nUpdated summary:"
)


class SummarizingMemory:
    def __init__(
        self,
        store: ConversationStore | None = None,
        router: Any = None,
        max_tokens: int = 2000,
        keep_recent: int = 8,
    ):
        self.store = store or get_store()
        self._router = router
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent

    @property
    def router(self) -> Any:
        if self._router is None:
            from agentic_studio.llm.router import get_router

            self._router = get_router()
        return self._router

    def append(self, thread_id: str, message: Message) -> None:
        self.store.append(thread_id, message)

    def load(self, thread_id: str) -> list[Message]:
        """Return a context-window-safe history, compacting first if needed."""
        history = self.store.history(thread_id)
        if not history:
            return []

        summary = self.store.get_summary(thread_id)
        total_tokens = sum(estimate_tokens(m.content) for m in history)
        if total_tokens <= self.max_tokens and summary is None:
            return history

        if total_tokens > self.max_tokens:
            self.compact(thread_id)
            summary = self.store.get_summary(thread_id)
            history = self.store.history(thread_id)

        messages: list[Message] = []
        if summary:
            messages.append(Message.system(f"Summary of earlier conversation:\n{summary[0]}"))
            messages.extend(history[-self.keep_recent :])
        else:
            messages = history
        return messages

    def compact(self, thread_id: str) -> str | None:
        """Fold everything older than `keep_recent` turns into the summary."""
        history = self.store.history(thread_id)
        if len(history) <= self.keep_recent:
            return None

        existing = self.store.get_summary(thread_id)
        previous = existing[0] if existing else "(none)"
        older = history[: -self.keep_recent]
        transcript = "\n".join(f"{m.role}: {m.content}" for m in older if m.content)

        with get_tracer().span("memory.compact", kind="chain", turns=len(older)):
            try:
                summary = self.router.complete(
                    SUMMARY_PROMPT.format(summary=previous, turns=transcript)
                ).strip()
            except Exception as exc:
                logger.warning("summarisation failed (%s); truncating instead", exc)
                summary = f"{previous}\n{transcript[:1000]}"

        self.store.set_summary(thread_id, summary, upto_seq=len(older))
        return summary

    def clear(self, thread_id: str) -> bool:
        return self.store.delete_thread(thread_id)
