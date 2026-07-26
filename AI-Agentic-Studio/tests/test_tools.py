"""Tool registry, schema inference, and the safety properties of each tool."""

from __future__ import annotations

from agentic_studio.agents.tools import REGISTRY, all_tools, default_tools
from agentic_studio.agents.tools.filesystem import resolve_in_sandbox
from agentic_studio.agents.tools.http import check_url
from agentic_studio.agents.tools.python_exec import BLOCKED_IMPORTS, execute
from agentic_studio.agents.tools.registry import ToolRegistry, infer_schema
from agentic_studio.agents.tools.sql import validate_query
from agentic_studio.agents.tools.web_search import run_search
from agentic_studio.core.types import ToolCall

# -- registry ---------------------------------------------------------------


def test_schema_is_inferred_from_hints_and_docstring():
    def sample(query: str, limit: int = 5, exact: bool = False) -> dict:
        """Search something.

        Args:
            query: What to search for.
            limit: How many results.
        """
        return {}

    schema = infer_schema(sample)

    assert schema["required"] == ["query"]
    assert schema["properties"]["limit"] == {"type": "integer", "description": "How many results.",
                                             "default": 5}
    assert schema["properties"]["exact"]["type"] == "boolean"
    assert schema["properties"]["query"]["description"] == "What to search for."


def test_optional_and_list_annotations_map_to_json_types():
    def sample(names: list[str], flag: bool | None = None) -> None:
        """Doc."""

    schema = infer_schema(sample)

    assert schema["properties"]["names"] == {"type": "array", "items": {"type": "string"}}
    assert schema["properties"]["flag"]["type"] == "boolean"


def test_decorator_registers_and_runs_a_tool():
    registry = ToolRegistry()

    @registry.tool(tags=("test",))
    def double(value: int) -> int:
        """Double a number."""
        return value * 2

    result = registry.run(ToolCall(name="double", arguments={"value": 21}))

    assert result.ok is True
    assert result.output == "42"
    assert registry.get("double").description == "Double a number."
    assert registry.specs(tags=["test"])[0].name == "double"


def test_unknown_tool_returns_a_failed_result():
    result = ToolRegistry().run(ToolCall(name="nope", arguments={}))

    assert result.ok is False
    assert "not registered" in result.error


def test_a_raising_tool_is_retried_then_reported():
    registry = ToolRegistry()
    attempts = {"count": 0}

    @registry.tool()
    def flaky() -> str:
        """Always fails."""
        attempts["count"] += 1
        raise RuntimeError("nope")

    result = registry.run(ToolCall(name="flaky", arguments={}), retries=2)

    assert attempts["count"] == 3
    assert result.ok is False
    assert "RuntimeError" in result.error


def test_a_hanging_tool_times_out():
    registry = ToolRegistry()

    @registry.tool()
    def slow() -> str:
        """Sleeps too long."""
        import time

        time.sleep(0.6)
        return "done"

    result = registry.run(ToolCall(name="slow", arguments={}), timeout_s=0.1, retries=0)

    assert result.ok is False
    assert "exceeded" in result.error


def test_run_many_preserves_order_when_parallel():
    registry = ToolRegistry()

    @registry.tool()
    def echo(value: str) -> str:
        """Echo the value."""
        import time

        time.sleep(0.05 if value == "first" else 0.0)
        return value

    calls = [ToolCall(name="echo", arguments={"value": v}) for v in ("first", "second", "third")]
    results = registry.run_many(calls, parallel=True)

    assert [result.output for result in results] == ["first", "second", "third"]


def test_dangerous_tools_are_excluded_from_the_default_set():
    default_names = {spec.name for spec in default_tools()}
    all_names = {spec.name for spec in all_tools()}

    assert "python_exec" not in default_names
    assert "write_file" not in default_names
    assert "python_exec" in all_names
    assert all(not spec.requires_approval for spec in default_tools())


def test_every_registered_tool_has_a_description_and_schema():
    for spec in all_tools():
        assert spec.description, f"{spec.name} is missing a description"
        assert spec.parameters.get("type") == "object", f"{spec.name} has a malformed schema"


# -- calculator + python sandbox --------------------------------------------


def test_calculator_evaluates_and_rejects_code():
    calculator = REGISTRY.get("calculator").func

    assert calculator("(1234 * 17) / 2") == "10489.0"
    assert "Error" in calculator("__import__('os').system('ls')")
    assert "Error" in calculator("open('x')")


def test_sandbox_runs_code_and_captures_stdout():
    result = execute("print(sum(range(10)))")

    assert result["ok"] is True
    assert result["stdout"].strip() == "45"


def test_sandbox_blocks_network_and_process_imports():
    for module in ("socket", "subprocess", "urllib"):
        assert module in BLOCKED_IMPORTS
        result = execute(f"import {module}")
        assert result["ok"] is False
        assert "not allowed" in result["stderr"]


def test_sandbox_enforces_a_timeout():
    result = execute("while True: pass", timeout_s=1.0)

    assert result["ok"] is False
    assert "exceeded" in result["stderr"]


# -- filesystem sandbox -----------------------------------------------------


def test_filesystem_paths_cannot_escape_the_sandbox():
    import pytest

    inside = resolve_in_sandbox("notes/a.txt")
    assert "sandbox" in str(inside)

    with pytest.raises(ValueError):
        resolve_in_sandbox("../../secrets.txt")


def test_write_then_read_then_delete_inside_the_sandbox():
    write = REGISTRY.get("write_file").func
    read = REGISTRY.get("read_file").func
    delete = REGISTRY.get("delete_file").func

    assert write("test-notes/hello.txt", "hybrid retrieval")["ok"] is True
    assert read("test-notes/hello.txt")["content"] == "hybrid retrieval"
    assert delete("test-notes/hello.txt")["deleted"] is True
    assert read("test-notes/hello.txt")["ok"] is False


def test_write_file_requires_approval():
    assert REGISTRY.get("write_file").requires_approval is True
    assert REGISTRY.get("read_file").requires_approval is False


# -- HTTP allowlist ---------------------------------------------------------


def test_http_allowlist_admits_the_host_and_its_subdomains(monkeypatch):
    # Stub DNS: the allowlist decision is what is under test, not resolution.
    monkeypatch.setattr("agentic_studio.agents.tools.http.is_private_address", lambda host: False)

    assert check_url("https://api.github.com/repos/x/y", ["api.github.com"])[0] is True
    assert check_url("https://sub.api.github.com/x", ["api.github.com"])[0] is True
    assert check_url("https://evil.example.com", ["api.github.com"])[0] is False
    assert check_url("https://notapi.github.com.attacker.io", ["api.github.com"])[0] is False


def test_http_denies_bad_schemes_localhost_and_an_empty_allowlist():
    assert check_url("file:///etc/passwd", ["api.github.com"])[0] is False
    assert check_url("http://localhost:8080", ["localhost"])[0] is False
    assert check_url("https://api.github.com", [])[0] is False, "empty allowlist denies everything"


def test_http_blocks_private_addresses():
    allowed, reason = check_url("http://127.0.0.1/x", ["127.0.0.1"])

    assert allowed is False
    assert "private" in reason


def test_http_request_refuses_a_non_allowlisted_host():
    outcome = REGISTRY.get("http_request").func("https://evil.example.com", "GET")

    assert outcome["ok"] is False
    assert "allowlist" in outcome["error"]


# -- SQL --------------------------------------------------------------------


def test_sql_accepts_reads_and_rejects_writes():
    import pytest

    assert validate_query("SELECT * FROM t") == "SELECT * FROM t"
    assert validate_query("WITH x AS (SELECT 1) SELECT * FROM x").startswith("WITH")

    for bad in ("DELETE FROM t", "DROP TABLE t", "UPDATE t SET a=1", "PRAGMA table_info(t)"):
        with pytest.raises(ValueError):
            validate_query(bad)


def test_sql_rejects_stacked_statements():
    import pytest

    with pytest.raises(ValueError):
        validate_query("SELECT 1; DROP TABLE t")


def test_sql_query_without_a_database_returns_an_error():
    result = REGISTRY.get("sql_query").func("SELECT 1")

    assert result["ok"] is False
    assert "no database configured" in result["error"]


# -- search + rag tools ----------------------------------------------------


def test_offline_search_falls_back_to_the_local_corpus(pipeline):
    outcome = run_search("what does BM25 catch", max_results=3, provider="offline")

    assert outcome["provider"] == "offline"
    assert outcome["count"] > 0
    assert outcome["results"][0]["url"].startswith("local://")


def test_rag_search_tool_returns_ranked_passages(pipeline):
    outcome = REGISTRY.get("rag_search").func("BM25 identifiers", top_k=2)

    assert outcome["count"] <= 2
    assert outcome["passages"]
    assert "rank" in outcome["passages"][0]


def test_corpus_stats_tool_reports_the_index(pipeline):
    stats = REGISTRY.get("corpus_stats").func()

    assert stats["chunks"] == 4
    assert stats["config"]["reranker"] == "lexical"


def test_list_sources_tool_groups_by_source(pipeline):
    outcome = REGISTRY.get("list_sources").func(limit=10)

    assert outcome["total_sources"] == 4
    assert outcome["sources"][0]["chunks"] >= 1
