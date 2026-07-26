"""Local GGUF inference through llama-cpp-python.

Loaded lazily so importing the studio never pulls in a native library.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from agentic_studio.core.errors import ProviderError
from agentic_studio.core.types import LLMResponse, Message, ToolSpec
from agentic_studio.llm.base import BaseProvider


class LlamaCppProvider(BaseProvider):
    name = "llamacpp"
    supports_tools = False
    supports_streaming = True

    def __init__(self, model: str, n_ctx: int = 4096, **kwargs: Any):
        super().__init__(model=model, **kwargs)
        self.n_ctx = n_ctx
        self._llm: Any = None

    def available(self) -> bool:
        if not Path(self.model).exists():
            return False
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self) -> Any:
        if self._llm is not None:
            return self._llm
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ProviderError(
                self.name,
                "llama-cpp-python is not installed. Run: uv sync --extra local-llm",
                retryable=False,
            ) from exc
        if not Path(self.model).exists():
            raise ProviderError(self.name, f"GGUF file not found: {self.model}", retryable=False)
        self._llm = Llama(model_path=self.model, n_ctx=self.n_ctx, verbose=False)
        return self._llm

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        llm = self._load()
        result = llm.create_chat_completion(
            messages=self._openai_style_messages(messages),
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        choice = (result.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content", "") or ""
        usage_data = result.get("usage", {})
        return LLMResponse(
            text=text,
            provider=self.name,
            model=Path(self.model).name,
            finish_reason=choice.get("finish_reason", "stop"),
            usage=self._usage(
                messages,
                text,
                prompt_tokens=usage_data.get("prompt_tokens"),
                completion_tokens=usage_data.get("completion_tokens"),
            ),
            raw=result,
        )

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        llm = self._load()
        stream = llm.create_chat_completion(
            messages=self._openai_style_messages(messages),
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )
        for event in stream:
            delta = ((event.get("choices") or [{}])[0]).get("delta", {})
            chunk = delta.get("content")
            if chunk:
                yield chunk
