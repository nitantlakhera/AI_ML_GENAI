"""OpenAI-compatible chat completions.

Works against OpenAI itself and any compatible gateway (vLLM, OpenRouter,
LM Studio, Azure-style proxies) by setting OPENAI_BASE_URL.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from agentic_studio.core.types import LLMResponse, Message, ToolCall, ToolSpec
from agentic_studio.llm.base import BaseProvider
from agentic_studio.llm.providers._http import post_json, post_sse

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIProvider(BaseProvider):
    name = "openai"
    supports_tools = True
    supports_vision = True
    supports_streaming = True

    def __init__(self, model: str, api_key: str = "", base_url: str | None = None, **kwargs: Any):
        super().__init__(model=model, **kwargs)
        self.api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")

    def available(self) -> bool:
        return bool(self.api_key) or self.base_url != DEFAULT_BASE_URL

    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, messages: list[Message], tools: list[ToolSpec] | None, **kwargs: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._openai_style_messages(messages),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        tool_payload = self._openai_style_tools(tools)
        if tool_payload:
            payload["tools"] = tool_payload
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")
        if kwargs.get("response_format"):
            payload["response_format"] = kwargs["response_format"]
        return payload

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        data = post_json(
            f"{self.base_url}/chat/completions",
            self._payload(messages, tools, **kwargs),
            self._headers(),
            self.timeout_s,
            self.name,
        )
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {})
        usage_data = data.get("usage", {})
        return LLMResponse(
            text=message.get("content") or "",
            provider=self.name,
            model=data.get("model", self.model),
            tool_calls=_parse_tool_calls(message.get("tool_calls")),
            finish_reason=choice.get("finish_reason", "stop"),
            usage=self._usage(
                messages,
                message.get("content") or "",
                prompt_tokens=usage_data.get("prompt_tokens"),
                completion_tokens=usage_data.get("completion_tokens"),
            ),
            raw=data,
        )

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        payload = self._payload(messages, kwargs.pop("tools", None), **kwargs)
        payload["stream"] = True
        for event in post_sse(
            f"{self.base_url}/chat/completions", payload, self._headers(), self.timeout_s, self.name
        ):
            delta = ((event.get("choices") or [{}])[0]).get("delta", {})
            chunk = delta.get("content")
            if chunk:
                yield chunk


def _parse_tool_calls(raw: list[dict[str, Any]] | None) -> list[ToolCall]:
    if not raw:
        return []
    calls: list[ToolCall] = []
    for entry in raw:
        function = entry.get("function", {})
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {"_raw": function.get("arguments", "")}
        calls.append(
            ToolCall(name=function.get("name", ""), arguments=arguments, id=entry.get("id") or "")
            if entry.get("id")
            else ToolCall(name=function.get("name", ""), arguments=arguments)
        )
    return calls
