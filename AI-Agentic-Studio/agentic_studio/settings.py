"""Central, typed configuration read from the environment.

Every default is chosen so `pytest` and the demo scripts work offline with no
API keys: the LLM chain defaults to the deterministic `echo` provider and
embeddings default to the dependency-free `hashing` backend.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:  # optional, keeps the package importable without python-dotenv
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent.parent


def _str(key: str, default: str) -> str:
    value = os.getenv(key)
    return default if value is None or value == "" else value


def _int(key: str, default: int) -> int:
    try:
        return int(_str(key, str(default)))
    except ValueError:
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(_str(key, str(default)))
    except ValueError:
        return default


def _bool(key: str, default: bool) -> bool:
    return _str(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _list(key: str, default: str) -> list[str]:
    raw = _str(key, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Paths:
    base: Path = BASE_DIR
    data_raw: Path = BASE_DIR / "data" / "raw"
    data_eval: Path = BASE_DIR / "data" / "eval"
    index: Path = BASE_DIR / "var" / "index"
    memory_db: Path = BASE_DIR / "var" / "memory.sqlite3"
    cache_db: Path = BASE_DIR / "var" / "cache.sqlite3"
    jobs_db: Path = BASE_DIR / "var" / "jobs.sqlite3"
    approvals_db: Path = BASE_DIR / "var" / "approvals.sqlite3"
    checkpoints_db: Path = BASE_DIR / "var" / "checkpoints.sqlite3"
    traces: Path = BASE_DIR / "var" / "traces.jsonl"
    reports: Path = BASE_DIR / "reports"
    tool_sandbox: Path = BASE_DIR / "var" / "sandbox"
    models: Path = BASE_DIR / "models"

    def ensure(self) -> None:
        for path in (
            self.data_raw,
            self.data_eval,
            self.index,
            self.reports,
            self.tool_sandbox,
            self.memory_db.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class LLMSettings:
    providers: list[str] = field(default_factory=lambda: _list("STUDIO_LLM_PROVIDERS", "echo"))
    temperature: float = _float("STUDIO_LLM_TEMPERATURE", 0.2)
    max_tokens: int = _int("STUDIO_LLM_MAX_TOKENS", 1024)
    timeout_s: float = _float("STUDIO_LLM_TIMEOUT_S", 60)
    max_retries: int = _int("STUDIO_LLM_MAX_RETRIES", 2)

    openai_api_key: str = _str("OPENAI_API_KEY", "")
    openai_model: str = _str("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url: str = _str("OPENAI_BASE_URL", "")

    anthropic_api_key: str = _str("ANTHROPIC_API_KEY", "")
    anthropic_model: str = _str("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    gemini_api_key: str = _str("GEMINI_API_KEY", "")
    gemini_model: str = _str("GEMINI_MODEL", "gemini-2.0-flash")

    ollama_base_url: str = _str("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = _str("OLLAMA_MODEL", "llama3.1")

    llamacpp_model_path: str = _str("LLAMACPP_MODEL_PATH", "models/llama-3-8b-instruct.Q4_K_M.gguf")
    llamacpp_n_ctx: int = _int("LLAMACPP_N_CTX", 4096)


@dataclass(frozen=True)
class CacheSettings:
    enabled: bool = _bool("STUDIO_CACHE_ENABLED", True)
    semantic: bool = _bool("STUDIO_CACHE_SEMANTIC", True)
    similarity_threshold: float = _float("STUDIO_CACHE_SIMILARITY_THRESHOLD", 0.95)
    ttl_s: int = _int("STUDIO_CACHE_TTL_S", 3600)


@dataclass(frozen=True)
class RetrievalSettings:
    embedding_backend: str = _str("STUDIO_EMBEDDING_BACKEND", "hashing")
    embedding_model: str = _str("STUDIO_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim: int = _int("STUDIO_EMBEDDING_DIM", 384)

    chunk_strategy: str = _str("STUDIO_CHUNK_STRATEGY", "recursive")
    chunk_size: int = _int("STUDIO_CHUNK_SIZE", 600)
    chunk_overlap: int = _int("STUDIO_CHUNK_OVERLAP", 80)
    parent_chunk_size: int = _int("STUDIO_PARENT_CHUNK_SIZE", 2400)

    top_k: int = _int("STUDIO_RETRIEVAL_TOP_K", 8)
    fetch_k: int = _int("STUDIO_RETRIEVAL_FETCH_K", 30)
    hybrid_enabled: bool = _bool("STUDIO_HYBRID_ENABLED", True)
    dense_weight: float = _float("STUDIO_HYBRID_DENSE_WEIGHT", 0.6)
    graph_rag_enabled: bool = _bool("STUDIO_GRAPH_RAG_ENABLED", True)

    reranker: str = _str("STUDIO_RERANKER", "lexical")
    rerank_model: str = _str("STUDIO_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    query_transform: str = _str("STUDIO_QUERY_TRANSFORM", "multi-query")
    query_variants: int = _int("STUDIO_QUERY_VARIANTS", 3)


@dataclass(frozen=True)
class AgentSettings:
    max_steps: int = _int("STUDIO_AGENT_MAX_STEPS", 12)
    parallel_tools: bool = _bool("STUDIO_AGENT_PARALLEL_TOOLS", True)
    tool_timeout_s: float = _float("STUDIO_AGENT_TOOL_TIMEOUT_S", 30)
    tool_retries: int = _int("STUDIO_AGENT_TOOL_RETRIES", 1)
    allowed_tools: list[str] = field(default_factory=lambda: _list("STUDIO_AGENT_ALLOWED_TOOLS", ""))
    hitl_enabled: bool = _bool("STUDIO_HITL_ENABLED", True)


@dataclass(frozen=True)
class ToolSettings:
    search_provider: str = _str("STUDIO_SEARCH_PROVIDER", "offline")
    tavily_api_key: str = _str("TAVILY_API_KEY", "")
    http_allowed_hosts: list[str] = field(
        default_factory=lambda: _list(
            "STUDIO_HTTP_ALLOWED_HOSTS", "api.github.com,jsonplaceholder.typicode.com"
        )
    )
    python_exec_timeout_s: float = _float("STUDIO_PYTHON_EXEC_TIMEOUT_S", 10)
    sql_database_url: str = _str("STUDIO_SQL_DATABASE_URL", "")


@dataclass(frozen=True)
class GuardrailSettings:
    enabled: bool = _bool("STUDIO_GUARDRAILS_ENABLED", True)
    pii_mode: str = _str("STUDIO_PII_MODE", "redact")
    moderation_mode: str = _str("STUDIO_MODERATION_MODE", "block")
    max_input_chars: int = _int("STUDIO_MAX_INPUT_CHARS", 20000)


@dataclass(frozen=True)
class ObservabilitySettings:
    tracing_enabled: bool = _bool("STUDIO_TRACING_ENABLED", True)
    trace_sink: str = _str("STUDIO_TRACE_SINK", "jsonl")
    log_level: str = _str("STUDIO_LOG_LEVEL", "INFO")
    langsmith_api_key: str = _str("LANGSMITH_API_KEY", "")
    langsmith_project: str = _str("LANGSMITH_PROJECT", "ai-agentic-studio")
    otel_endpoint: str = _str("OTEL_EXPORTER_OTLP_ENDPOINT", "")


@dataclass(frozen=True)
class APISettings:
    host: str = _str("STUDIO_API_HOST", "0.0.0.0")
    port: int = _int("STUDIO_API_PORT", 8100)
    api_keys: list[str] = field(default_factory=lambda: _list("STUDIO_API_KEYS", ""))
    rate_limit_per_minute: int = _int("STUDIO_RATE_LIMIT_PER_MINUTE", 60)
    cors_origins: list[str] = field(default_factory=lambda: _list("STUDIO_CORS_ORIGINS", "*"))

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)
    llm: LLMSettings = field(default_factory=LLMSettings)
    cache: CacheSettings = field(default_factory=CacheSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    tools: ToolSettings = field(default_factory=ToolSettings)
    guardrails: GuardrailSettings = field(default_factory=GuardrailSettings)
    observability: ObservabilitySettings = field(default_factory=ObservabilitySettings)
    api: APISettings = field(default_factory=APISettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.paths.ensure()
    return settings


def reload_settings() -> Settings:
    """Re-read the environment. Used by tests that patch env vars."""
    get_settings.cache_clear()
    return get_settings()
