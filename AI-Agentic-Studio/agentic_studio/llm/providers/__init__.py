"""Provider registry.

`build_provider(name)` returns a configured provider or raises ConfigError.
Adding a vendor means adding one module here - nothing else changes.
"""

from __future__ import annotations

from agentic_studio.core.errors import ConfigError
from agentic_studio.llm.base import BaseProvider
from agentic_studio.llm.providers.anthropic_provider import AnthropicProvider
from agentic_studio.llm.providers.echo_provider import EchoProvider
from agentic_studio.llm.providers.gemini_provider import GeminiProvider
from agentic_studio.llm.providers.llamacpp_provider import LlamaCppProvider
from agentic_studio.llm.providers.ollama_provider import OllamaProvider
from agentic_studio.llm.providers.openai_provider import OpenAIProvider
from agentic_studio.llm.providers.scripted_provider import ScriptedProvider
from agentic_studio.settings import get_settings

PROVIDERS: dict[str, type[BaseProvider]] = {
    "echo": EchoProvider,
    "scripted": ScriptedProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
    "llamacpp": LlamaCppProvider,
}


def build_provider(name: str) -> BaseProvider:
    key = name.strip().lower()
    provider_cls = PROVIDERS.get(key)
    if provider_cls is None:
        raise ConfigError(f"unknown LLM provider '{name}'. Known: {', '.join(sorted(PROVIDERS))}")

    llm = get_settings().llm
    common = {
        "temperature": llm.temperature,
        "max_tokens": llm.max_tokens,
        "timeout_s": llm.timeout_s,
    }

    if key == "openai":
        return OpenAIProvider(model=llm.openai_model, api_key=llm.openai_api_key,
                              base_url=llm.openai_base_url or None, **common)
    if key == "anthropic":
        return AnthropicProvider(model=llm.anthropic_model, api_key=llm.anthropic_api_key, **common)
    if key == "gemini":
        return GeminiProvider(model=llm.gemini_model, api_key=llm.gemini_api_key, **common)
    if key == "ollama":
        return OllamaProvider(model=llm.ollama_model, base_url=llm.ollama_base_url, **common)
    if key == "llamacpp":
        return LlamaCppProvider(model=llm.llamacpp_model_path, n_ctx=llm.llamacpp_n_ctx, **common)
    return provider_cls(model=key, **common)


__all__ = [
    "AnthropicProvider",
    "EchoProvider",
    "GeminiProvider",
    "LlamaCppProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "PROVIDERS",
    "ScriptedProvider",
    "build_provider",
]
