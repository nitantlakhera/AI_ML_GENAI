"""Anthropic Messages API provider."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from agentic_studio.core.types import LLMResponse, Message, ToolCall, ToolSpec
from agentic_studio.llm.base import BaseProvider
from agentic_studio.llm.providers._http import post_json, post_sse

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    name = "anthropic"
    supports_tools = True
    supports_vision = True
    supports_streaming = True

    def __init__(self, model: str, api_key: str = "", **kwargs: Any):
        super().__init__(model=model, **kwargs)
        self.api_key = api_key

    def available(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key, "anthropic-version": API_VERSION}

    def _payload(self, messages: list[Message], tools: list[ToolSpec] | None, **kwargs: Any) -> dict[str, Any]:
        system_parts = [m.content for m in messages if m.role == "system" and m.content]
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "messages": _convert(messages),
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
        return payload

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        data = post_json(API_URL, self._payload(messages, tools, **kwargs), self._headers(),
                         self.timeout_s, self.name)
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(name=block.get("name", ""), arguments=block.get("input", {})))

        text = "".join(text_parts)
        usage_data = data.get("usage", {})
        return LLMResponse(
            text=text,
            provider=self.name,
            model=data.get("model", self.model),
            tool_calls=tool_calls,
            finish_reason=data.get("stop_reason", "stop"),
            usage=self._usage(
                messages,
                text,
                prompt_tokens=usage_data.get("input_tokens"),
                completion_tokens=usage_data.get("output_tokens"),
            ),
            raw=data,
        )

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        payload = self._payload(messages, kwargs.pop("tools", None), **kwargs)
        payload["stream"] = True
        for event in post_sse(API_URL, payload, self._headers(), self.timeout_s, self.name):
            if event.get("type") == "content_block_delta":
                chunk = (event.get("delta") or {}).get("text")
                if chunk:
                    yield chunk


def _convert(messages: list[Message]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id or "",
                            "content": message.content,
                        }
                    ],
                }
            )
            continue
        if message.tool_calls:
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            for call in message.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                )
            converted.append({"role": "assistant", "content": blocks})
            continue
        if message.images and message.role == "user":
            blocks = [{"type": "text", "text": message.content}]
            for image in message.images:
                blocks.append({"type": "image", "source": {"type": "url", "url": image}})
            converted.append({"role": "user", "content": blocks})
            continue
        converted.append({"role": message.role, "content": message.content})
    return converted
