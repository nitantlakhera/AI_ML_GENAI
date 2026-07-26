"""Google Gemini generateContent provider."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from agentic_studio.core.types import LLMResponse, Message, ToolCall, ToolSpec
from agentic_studio.llm.base import BaseProvider
from agentic_studio.llm.providers._http import post_json

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider(BaseProvider):
    name = "gemini"
    supports_tools = True
    supports_vision = True
    supports_streaming = False

    def __init__(self, model: str, api_key: str = "", **kwargs: Any):
        super().__init__(model=model, **kwargs)
        self.api_key = api_key

    def available(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "contents": _convert(messages),
            "generationConfig": {
                "temperature": kwargs.get("temperature", self.temperature),
                "maxOutputTokens": kwargs.get("max_tokens", self.max_tokens),
            },
        }
        system_parts = [m.content for m in messages if m.role == "system" and m.content]
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        if tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {"name": t.name, "description": t.description, "parameters": t.parameters}
                        for t in tools
                    ]
                }
            ]

        url = f"{BASE_URL}/{self.model}:generateContent?key={self.api_key}"
        data = post_json(url, payload, {}, self.timeout_s, self.name)

        candidate = (data.get("candidates") or [{}])[0]
        parts = (candidate.get("content") or {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts if "text" in part)
        tool_calls = [
            ToolCall(name=part["functionCall"].get("name", ""), arguments=part["functionCall"].get("args", {}))
            for part in parts
            if "functionCall" in part
        ]
        usage_data = data.get("usageMetadata", {})
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            tool_calls=tool_calls,
            finish_reason=candidate.get("finishReason", "stop"),
            usage=self._usage(
                messages,
                text,
                prompt_tokens=usage_data.get("promptTokenCount"),
                completion_tokens=usage_data.get("candidatesTokenCount"),
            ),
            raw=data,
        )

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        yield self.generate(messages, **kwargs).text


def _convert(messages: list[Message]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue
        role = "model" if message.role == "assistant" else "user"
        parts: list[dict[str, Any]] = []
        if message.role == "tool":
            parts.append(
                {
                    "functionResponse": {
                        "name": message.name or "tool",
                        "response": {"result": message.content},
                    }
                }
            )
        else:
            if message.content:
                parts.append({"text": message.content})
            for call in message.tool_calls:
                parts.append({"functionCall": {"name": call.name, "args": call.arguments}})
            for image in message.images:
                parts.append({"fileData": {"fileUri": image}})
        if parts:
            contents.append({"role": role, "parts": parts})
    return contents
