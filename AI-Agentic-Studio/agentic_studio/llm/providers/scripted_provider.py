"""A provider that replays a fixed script.

Used by tests and demos to drive an agent through an exact sequence of tool
calls and final answers, making agent behaviour assertable without a network.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from agentic_studio.core.types import LLMResponse, Message, ToolCall, ToolSpec
from agentic_studio.llm.base import BaseProvider

Step = LLMResponse | str | dict[str, Any]


class ScriptedProvider(BaseProvider):
    name = "scripted"
    supports_tools = True
    supports_streaming = True

    def __init__(self, model: str = "scripted", script: Sequence[Step] | None = None, **kwargs: Any):
        super().__init__(model=model or "scripted", **kwargs)
        self.script: list[Step] = list(script or [])
        self.calls: list[list[Message]] = []
        self._cursor = 0

    def push(self, step: Step) -> ScriptedProvider:
        self.script.append(step)
        return self

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if self._cursor >= len(self.script):
            text = "scripted provider exhausted"
            return LLMResponse(text=text, provider=self.name, model=self.model,
                               usage=self._usage(messages, text))

        step = self.script[self._cursor]
        self._cursor += 1
        return self._to_response(step, messages)

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        yield self.generate(messages, **kwargs).text

    def reset(self) -> None:
        self._cursor = 0
        self.calls.clear()

    def _to_response(self, step: Step, messages: list[Message]) -> LLMResponse:
        if isinstance(step, LLMResponse):
            return step
        if isinstance(step, str):
            return LLMResponse(text=step, provider=self.name, model=self.model,
                               usage=self._usage(messages, step))

        tool_calls = [
            ToolCall(name=call["name"], arguments=call.get("arguments", {}))
            for call in step.get("tool_calls", [])
        ]
        text = step.get("text", "")
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            usage=self._usage(messages, text),
        )
