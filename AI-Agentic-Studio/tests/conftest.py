"""Shared fixtures.

Every fixture points state at a tmp_path, so tests never touch the developer's
real index, memory, or checkpoints, and they run in any order.
"""

from __future__ import annotations

import pytest

from agentic_studio.agents.checkpoint import MemoryCheckpointer
from agentic_studio.agents.hitl import ApprovalStore, reset_approval_store, set_approval_store
from agentic_studio.api.jobs import JobStore, reset_job_store
from agentic_studio.core.types import Chunk
from agentic_studio.guardrails.policy import GuardrailPolicy, reset_policy, set_policy
from agentic_studio.llm.providers.echo_provider import EchoProvider
from agentic_studio.llm.providers.scripted_provider import ScriptedProvider
from agentic_studio.llm.router import LLMRouter, reset_router, set_router
from agentic_studio.memory.store import ConversationStore, reset_store, set_store
from agentic_studio.observability.metrics import METRICS
from agentic_studio.observability.tracing import Tracer, set_tracer
from agentic_studio.rag.pipeline import RagConfig, RagPipeline, reset_pipeline, set_pipeline
from agentic_studio.rag.vector_store import NumpyVectorStore
from agentic_studio.settings import get_settings

# Paths the studio writes to. Redirecting them per test keeps the sandbox, cache,
# and traces of one test out of the next one - and out of the repo.
_REDIRECTED_PATHS = {
    "tool_sandbox": "sandbox",
    "cache_db": "cache.sqlite3",
    "checkpoints_db": "checkpoints.sqlite3",
    "traces": "traces.jsonl",
    "reports": "reports",
    "index": "index",
}

CORPUS = [
    "Reciprocal rank fusion merges rankings from multiple retrievers without needing their "
    "scores to be comparable. It uses only the rank of each document.",
    "BM25 is a lexical ranking function. BM25 catches exact identifiers and rare terms that "
    "dense retrieval misses.",
    "A cross-encoder reranker scores the query and document together, which improves precision "
    "at the top of the ranking.",
    "Human in the loop means an agent pauses for approval before running a tool with side "
    "effects, such as writing files or executing code.",
]


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    """Reset every singleton and redirect persistent state into tmp_path."""
    paths = get_settings().paths
    originals = {name: getattr(paths, name) for name in _REDIRECTED_PATHS}
    for name, leaf in _REDIRECTED_PATHS.items():
        object.__setattr__(paths, name, tmp_path / leaf)  # Paths is frozen by design
    (tmp_path / "sandbox").mkdir()

    METRICS.reset()
    set_tracer(Tracer(sink="memory", enabled=True))
    set_store(ConversationStore(path=tmp_path / "memory.sqlite3"))
    set_approval_store(ApprovalStore(path=tmp_path / "approvals.sqlite3"))
    monkeypatch.setattr("agentic_studio.api.jobs._STORE", JobStore(path=tmp_path / "jobs.sqlite3"))
    set_policy(GuardrailPolicy())
    yield
    reset_router()
    reset_pipeline()
    reset_store()
    reset_approval_store()
    reset_job_store()
    reset_policy()
    for name, value in originals.items():
        object.__setattr__(paths, name, value)


@pytest.fixture
def echo_router() -> LLMRouter:
    router = LLMRouter(providers=[EchoProvider()], use_cache=False)
    set_router(router)
    return router


@pytest.fixture
def scripted() -> ScriptedProvider:
    """A provider whose responses the test controls exactly."""
    provider = ScriptedProvider()
    set_router(LLMRouter(providers=[provider], use_cache=False))
    return provider


@pytest.fixture
def store(tmp_path) -> NumpyVectorStore:
    vector_store = NumpyVectorStore(path=tmp_path / "index", autoload=False)
    vector_store.upsert(
        [
            Chunk(text=text, id=f"c{index}", metadata={"source": f"doc{index}.md", "page": index + 1})
            for index, text in enumerate(CORPUS)
        ]
    )
    return vector_store


@pytest.fixture
def pipeline(store, echo_router) -> RagPipeline:
    config = RagConfig(top_k=4, fetch_k=8, query_transform="none", reranker="lexical")
    built = RagPipeline(store=store, config=config, router=echo_router)
    set_pipeline(built)
    return built


@pytest.fixture
def checkpointer() -> MemoryCheckpointer:
    return MemoryCheckpointer()
