"""Ollama provider for fully local models with tool-calling support."""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from agentic_studio.core.types import LLMResponse, Message, ToolCall, ToolSpec
from agentic_studio.llm.base import BaseProvider
from agentic_studio.llm.providers._http import post_json, post_jsonlines


class OllamaProvider(BaseProvider):
    name = "ollama"
    supports_tools = True
    supports_vision = True
    supports_streaming = True

    def __init__(self, model: str, base_url: str = "http://localhost:11434", **kwargs: Any):
        super().__init__(model=model, **kwargs)
        self.base_url = base_url.rstrip("/")

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=1.5) as response:
                return response.status == 200
        except Exception:
            return False

    def _payload(self, messages: list[Message], tools: list[ToolSpec] | None, **kwargs: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._openai_style_messages(messages),
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
            "stream": False,
        }
        tool_payload = self._openai_style_tools(tools)
        if tool_payload:
            payload["tools"] = tool_payload
        return payload

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        data = post_json(f"{self.base_url}/api/chat", self._payload(messages, tools, **kwargs),
                         {}, self.timeout_s, self.name)
        message = data.get("message", {})
        text = message.get("content", "") or ""
        tool_calls = [
            ToolCall(name=(c.get("function") or {}).get("name", ""),
                     arguments=(c.get("function") or {}).get("arguments", {}) or {})
            for c in message.get("tool_calls", [])
        ]
        return LLMResponse(
            text=text,
            provider=self.name,
            model=data.get("model", self.model),
            tool_calls=tool_calls,
            usage=self._usage(
                messages,
                text,
                prompt_tokens=data.get("prompt_eval_count"),
                completion_tokens=data.get("eval_count"),
            ),
            raw=data,
        )

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        payload = self._payload(messages, kwargs.pop("tools", None), **kwargs)
        payload["stream"] = True
        for event in post_jsonlines(f"{self.base_url}/api/chat", payload, {}, self.timeout_s, self.name):
            chunk = (event.get("message") or {}).get("content")
            if chunk:
                yield chunk
            if event.get("done"):
                return
