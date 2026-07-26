"""LLM routing, failover, caching, streaming, and structured output."""

from __future__ import annotations

import pytest

from agentic_studio.core.errors import AllProvidersFailed, ProviderError
from agentic_studio.core.types import LLMResponse, Message, ToolSpec
from agentic_studio.llm.base import BaseProvider
from agentic_studio.llm.cache import ResponseCache
from agentic_studio.llm.providers import build_provider
from agentic_studio.llm.providers.echo_provider import EchoProvider
from agentic_studio.llm.router import LLMRouter
from agentic_studio.llm.structured import (
    StructuredOutputError,
    extract_json,
    generate_structured,
    json_schema_of,
)
from agentic_studio.observability.metrics import METRICS


class AlwaysFails(BaseProvider):
    name = "always_fails"

    def __init__(self, retryable: bool = True):
        super().__init__(model="broken")
        self.attempts = 0
        self.retryable = retryable

    def generate(self, messages, tools=None, **kwargs):
        self.attempts += 1
        raise ProviderError(self.name, "boom", retryable=self.retryable)


class CountingProvider(BaseProvider):
    name = "counting"
    supports_streaming = True

    def __init__(self):
        super().__init__(model="counting")
        self.calls = 0

    def generate(self, messages, tools=None, **kwargs):
        self.calls += 1
        return LLMResponse(text=f"call {self.calls}", provider=self.name, model=self.model,
                           usage=self._usage(messages, "x"))


def test_router_fails_over_to_next_provider():
    broken = AlwaysFails()
    router = LLMRouter(providers=[broken, EchoProvider()], max_retries=1, use_cache=False)

    response = router.generate([Message.user("hello")])

    assert response.provider == "echo"
    assert broken.attempts == 2, "should retry once before failing over"


def test_router_does_not_retry_non_retryable_errors():
    broken = AlwaysFails(retryable=False)
    router = LLMRouter(providers=[broken, EchoProvider()], max_retries=3, use_cache=False)

    router.generate([Message.user("hello")])

    assert broken.attempts == 1


def test_router_raises_when_every_provider_fails():
    router = LLMRouter(providers=[AlwaysFails(), AlwaysFails()], max_retries=0, use_cache=False)

    with pytest.raises(AllProvidersFailed) as excinfo:
        router.generate([Message.user("hello")])

    assert "always_fails" in str(excinfo.value)


def test_exact_cache_prevents_a_second_provider_call(tmp_path):
    provider = CountingProvider()
    cache = ResponseCache(path=tmp_path / "cache.sqlite3", semantic=False)
    router = LLMRouter(providers=[provider], cache=cache, use_cache=True)
    messages = [Message.user("what is hybrid retrieval")]

    first = router.generate(messages)
    second = router.generate(messages)

    assert provider.calls == 1
    assert second.cached is True
    assert second.text == first.text
    assert METRICS.counter("llm_cache_hits", provider="counting") == 1


def test_semantic_cache_matches_a_paraphrase(tmp_path):
    provider = CountingProvider()
    cache = ResponseCache(path=tmp_path / "cache.sqlite3", semantic=True, similarity_threshold=0.75)
    router = LLMRouter(providers=[provider], cache=cache, use_cache=True)

    router.generate([Message.user("explain reciprocal rank fusion")])
    hit = router.generate([Message.user("explain reciprocal rank fusion please")])

    assert provider.calls == 1
    assert hit.cached is True


def test_tool_calling_turns_are_never_cached(tmp_path):
    provider = CountingProvider()
    cache = ResponseCache(path=tmp_path / "cache.sqlite3", semantic=False)
    router = LLMRouter(providers=[provider], cache=cache, use_cache=True)
    spec = ToolSpec(name="noop", description="n", parameters={"type": "object", "properties": {}},
                    func=lambda: "")

    router.generate([Message.user("go")], tools=[spec])
    router.generate([Message.user("go")], tools=[spec])

    assert provider.calls == 2


def test_streaming_yields_multiple_chunks(echo_router):
    chunks = list(echo_router.stream([Message.user("describe hybrid retrieval briefly")]))

    assert len(chunks) > 1
    assert "".join(chunks).strip()


def test_metrics_record_tokens_and_latency(echo_router):
    echo_router.generate([Message.user("hello world")])

    snapshot = METRICS.snapshot()
    assert snapshot["counters"]["llm_calls{provider=echo}"] == 1
    assert snapshot["counters"]["llm_prompt_tokens{provider=echo}"] > 0
    assert snapshot["histograms"]["llm_latency_ms{provider=echo}"]["count"] == 1


def test_echo_answers_extractively_from_a_context_block(echo_router):
    prompt = (
        "Context:\n[1] (source: a.md)\nBM25 catches exact identifiers that dense retrieval misses.\n\n"
        "[2] (source: b.md)\nCross-encoders improve precision.\n\n"
        "Question: What does BM25 catch?\n\nAnswer with citations:"
    )
    text = echo_router.complete(prompt)

    assert "identifiers" in text
    assert "[1]" in text


def test_echo_selects_a_tool_from_the_prompt():
    spec = ToolSpec(
        name="web_search",
        description="search",
        parameters={"type": "object", "properties": {"query": {"type": "string"}},
                    "required": ["query"]},
        func=lambda query: "",
    )
    response = EchoProvider().generate([Message.user("search for the latest news")], tools=[spec])

    assert [call.name for call in response.tool_calls] == ["web_search"]
    assert response.tool_calls[0].arguments["query"]


def test_echo_stops_calling_tools_once_results_exist():
    spec = ToolSpec(
        name="web_search", description="search",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        func=lambda query="": "",
    )
    messages = [
        Message.user("search for the latest news"),
        Message(role="tool", content="found three articles about retrieval.", name="web_search",
                tool_call_id="1"),
    ]
    response = EchoProvider().generate(messages, tools=[spec])

    assert response.tool_calls == []
    assert response.text


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": 2} hope that helps') == {"a": 2}
    with pytest.raises(StructuredOutputError):
        extract_json("no json at all")


def test_structured_output_validates_against_a_schema(echo_router):
    schema = {
        "type": "object",
        "properties": {"title": {"type": "string"}, "score": {"type": "number"}},
        "required": ["title", "score"],
    }
    parsed = generate_structured("summarise this", schema, router=echo_router)

    assert set(parsed) == {"title", "score"}


def test_json_schema_of_a_dataclass():
    from dataclasses import dataclass

    @dataclass
    class Finding:
        claim: str
        confidence: float
        supported: bool = True

    schema = json_schema_of(Finding)

    assert schema["properties"]["confidence"]["type"] == "number"
    assert schema["required"] == ["claim", "confidence"]


def test_build_provider_rejects_an_unknown_name():
    from agentic_studio.core.errors import ConfigError

    with pytest.raises(ConfigError):
        build_provider("not-a-provider")


def test_unconfigured_hosted_providers_report_unavailable():
    assert build_provider("openai").available() is False
    assert build_provider("anthropic").available() is False
    assert build_provider("echo").available() is True
