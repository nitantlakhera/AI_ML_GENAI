"""The provider contract.

A provider translates `list[Message]` into an `LLMResponse`. Nothing above this
layer knows which vendor answered.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from agentic_studio.core.types import LLMResponse, Message, ToolSpec
from agentic_studio.observability.metrics import estimate_cost, estimate_tokens


class BaseProvider(ABC):
    name: str = "base"
    supports_tools: bool = False
    supports_vision: bool = False
    supports_streaming: bool = False

    def __init__(self, model: str, temperature: float = 0.2, max_tokens: int = 1024, timeout_s: float = 60.0):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def available(self) -> bool:
        """Whether this provider can be used right now (keys present, SDK importable)."""
        return True

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        ...

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        """Default streaming: emit the completed text in one chunk."""
        yield self.generate(messages, **kwargs).text

    # -- helpers shared by concrete providers -------------------------------

    def _usage(self, messages: list[Message], output: str, prompt_tokens: int | None = None,
               completion_tokens: int | None = None):
        from agentic_studio.core.types import Usage

        prompt = prompt_tokens if prompt_tokens is not None else sum(
            estimate_tokens(m.content) for m in messages
        )
        completion = completion_tokens if completion_tokens is not None else estimate_tokens(output)
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            cost_usd=estimate_cost(self.model, prompt, completion),
        )

    @staticmethod
    def _openai_style_messages(messages: list[Message]) -> list[dict[str, Any]]:
        """Convert to the role/content shape used by OpenAI-compatible APIs."""
        payload: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "tool":
                payload.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id or "",
                        "content": message.content,
                    }
                )
                continue

            if message.images and message.role == "user":
                parts: list[dict[str, Any]] = [{"type": "text", "text": message.content}]
                for image in message.images:
                    parts.append({"type": "image_url", "image_url": {"url": image}})
                payload.append({"role": "user", "content": parts})
                continue

            entry: dict[str, Any] = {"role": message.role, "content": message.content}
            if message.tool_calls:
                import json

                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    }
                    for call in message.tool_calls
                ]
            payload.append(entry)
        return payload

    @staticmethod
    def _openai_style_tools(tools: list[ToolSpec] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(model={self.model!r})"
