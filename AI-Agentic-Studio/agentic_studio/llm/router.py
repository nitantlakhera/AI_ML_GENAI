"""The single entry point for every model call in the studio.

Responsibilities:
  * provider chain with automatic failover (`STUDIO_LLM_PROVIDERS=openai,ollama,echo`)
  * bounded retries with exponential backoff on retryable provider errors
  * exact + semantic response caching
  * token, cost, latency, and error metrics
  * one trace span per call, nested under whatever invoked it
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from typing import Any

from agentic_studio.core.errors import AllProvidersFailed, ProviderError
from agentic_studio.core.types import LLMResponse, Message, ToolSpec
from agentic_studio.llm.base import BaseProvider
from agentic_studio.llm.cache import ResponseCache
from agentic_studio.llm.providers import build_provider
from agentic_studio.observability.logs import get_logger
from agentic_studio.observability.metrics import METRICS
from agentic_studio.observability.tracing import get_tracer
from agentic_studio.settings import get_settings

logger = get_logger("llm.router")


class LLMRouter:
    def __init__(
        self,
        providers: Sequence[BaseProvider] | None = None,
        cache: ResponseCache | None = None,
        max_retries: int | None = None,
        use_cache: bool | None = None,
    ):
        settings = get_settings()
        self.providers: list[BaseProvider] = list(providers) if providers else _from_settings()
        self.max_retries = settings.llm.max_retries if max_retries is None else max_retries
        self.use_cache = settings.cache.enabled if use_cache is None else use_cache
        self._cache = cache
        if self.use_cache and self._cache is None:
            self._cache = ResponseCache()

    @property
    def primary(self) -> BaseProvider:
        return self.providers[0]

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "provider": p.name,
                "model": p.model,
                "available": p.available(),
                "tools": p.supports_tools,
                "vision": p.supports_vision,
                "streaming": p.supports_streaming,
            }
            for p in self.providers
        ]

    # -- generation ---------------------------------------------------------

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        caching = self.use_cache if use_cache is None else use_cache
        # Tool-calling turns are never cached: identical prompts can legitimately
        # need different tool arguments as the conversation state changes.
        caching = caching and not tools
        candidates = self._candidates(require_tools=bool(tools))
        failures: dict[str, str] = {}

        with get_tracer().span("llm.generate", kind="llm", messages=len(messages),
                               tools=[t.name for t in tools or []]) as span:
            for provider in candidates:
                if caching and self._cache is not None:
                    hit = self._cache.get(messages, provider.model, **_cache_options(kwargs))
                    if hit is not None:
                        METRICS.incr("llm_cache_hits", provider=provider.name)
                        span.set(provider=provider.name, cached=True)
                        return hit

                started = time.perf_counter()
                for attempt in range(self.max_retries + 1):
                    try:
                        response = provider.generate(messages, tools=tools, **kwargs)
                    except ProviderError as exc:
                        failures[provider.name] = str(exc)
                        METRICS.incr("llm_errors", provider=provider.name)
                        if not exc.retryable or attempt == self.max_retries:
                            break
                        time.sleep(min(2**attempt * 0.25, 4.0))
                        continue
                    except Exception as exc:  # unexpected provider bug
                        failures[provider.name] = f"{type(exc).__name__}: {exc}"
                        METRICS.incr("llm_errors", provider=provider.name)
                        break

                    response.latency_ms = (time.perf_counter() - started) * 1000
                    self._record(provider, response, span)
                    if caching and self._cache is not None:
                        self._cache.set(messages, provider.model, response, **_cache_options(kwargs))
                    return response

                logger.warning("provider failed, failing over: %s", provider.name)

            span.set(failures=failures)
            raise AllProvidersFailed(failures)

    def stream(
        self,
        messages: list[Message],
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield text chunks, falling back to a single chunk for non-streaming providers."""
        caching = self.use_cache if use_cache is None else use_cache
        candidates = self._candidates(require_streaming=False)
        failures: dict[str, str] = {}

        with get_tracer().span("llm.stream", kind="llm", messages=len(messages)) as span:
            if caching and self._cache is not None:
                for provider in candidates:
                    hit = self._cache.get(messages, provider.model, **_cache_options(kwargs))
                    if hit is not None:
                        METRICS.incr("llm_cache_hits", provider=provider.name)
                        span.set(provider=provider.name, cached=True)
                        yield hit.text
                        return

            for provider in candidates:
                started = time.perf_counter()
                collected: list[str] = []
                try:
                    for chunk in provider.stream(messages, **kwargs):
                        collected.append(chunk)
                        yield chunk
                except (ProviderError, Exception) as exc:  # noqa: B014 - explicit for clarity
                    failures[provider.name] = f"{type(exc).__name__}: {exc}"
                    METRICS.incr("llm_errors", provider=provider.name)
                    if collected:
                        # Already emitted partial output; do not restart on another provider.
                        return
                    continue

                text = "".join(collected)
                response = LLMResponse(
                    text=text,
                    provider=provider.name,
                    model=provider.model,
                    usage=provider._usage(messages, text),
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
                self._record(provider, response, span)
                if caching and self._cache is not None:
                    self._cache.set(messages, provider.model, response, **_cache_options(kwargs))
                return

            span.set(failures=failures)
            raise AllProvidersFailed(failures)

    def complete(self, prompt: str, system: str | None = None, **kwargs: Any) -> str:
        messages: list[Message] = []
        if system:
            messages.append(Message.system(system))
        messages.append(Message.user(prompt))
        return self.generate(messages, **kwargs).text

    # -- internals ----------------------------------------------------------

    def _candidates(self, require_tools: bool = False, require_streaming: bool = False) -> list[BaseProvider]:
        usable = [p for p in self.providers if p.available()]
        if not usable:
            usable = list(self.providers)
        if require_tools:
            preferred = [p for p in usable if p.supports_tools]
            if preferred:
                return preferred
        if require_streaming:
            preferred = [p for p in usable if p.supports_streaming]
            if preferred:
                return preferred
        return usable

    def _record(self, provider: BaseProvider, response: LLMResponse, span: Any) -> None:
        METRICS.incr("llm_calls", provider=provider.name)
        METRICS.incr("llm_prompt_tokens", response.usage.prompt_tokens, provider=provider.name)
        METRICS.incr("llm_completion_tokens", response.usage.completion_tokens, provider=provider.name)
        METRICS.incr("llm_cost_usd", response.usage.cost_usd, provider=provider.name)
        METRICS.observe("llm_latency_ms", response.latency_ms, provider=provider.name)
        span.set(
            provider=provider.name,
            model=response.model,
            tokens=response.usage.total_tokens,
            cost_usd=round(response.usage.cost_usd, 6),
            tool_calls=[c.name for c in response.tool_calls],
        )


def _cache_options(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "temperature": kwargs.get("temperature"),
        "max_tokens": kwargs.get("max_tokens"),
        "response_format": kwargs.get("response_format"),
    }


def _from_settings() -> list[BaseProvider]:
    names = get_settings().llm.providers or ["echo"]
    providers: list[BaseProvider] = []
    for name in names:
        try:
            providers.append(build_provider(name))
        except Exception as exc:
            logger.warning("skipping provider %s: %s", name, exc)
    if not providers:
        providers.append(build_provider("echo"))
    # Guarantee the chain always terminates in something that works offline.
    if not any(p.name == "echo" for p in providers):
        providers.append(build_provider("echo"))
    return providers


_ROUTER: LLMRouter | None = None


def get_router() -> LLMRouter:
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = LLMRouter()
    return _ROUTER


def set_router(router: LLMRouter) -> None:
    global _ROUTER
    _ROUTER = router


def reset_router() -> None:
    global _ROUTER
    _ROUTER = None
