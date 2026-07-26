"""Read-only SQL access over SQLite.

Only single SELECT/WITH statements are accepted, results are row-capped, and the
connection is opened in immutable mode so a mistake cannot mutate data.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from agentic_studio.agents.tools.registry import tool
from agentic_studio.settings import get_settings

MAX_ROWS = 200

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|pragma|vacuum|reindex)\b",
    re.I,
)


def resolve_database(database: str | None = None) -> Path:
    configured = database or get_settings().tools.sql_database_url
    if not configured:
        raise ValueError("no database configured; set STUDIO_SQL_DATABASE_URL or pass `database`")
    path = Path(configured.replace("sqlite:///", "").replace("sqlite://", ""))
    if not path.exists():
        raise FileNotFoundError(f"database not found: {path}")
    return path


def validate_query(query: str) -> str:
    cleaned = query.strip().rstrip(";").strip()
    if not cleaned:
        raise ValueError("empty query")
    if ";" in cleaned:
        raise ValueError("only a single statement is allowed")
    if not re.match(r"^\s*(select|with)\b", cleaned, re.I):
        raise ValueError("only SELECT and WITH queries are allowed")
    if _FORBIDDEN.search(cleaned):
        raise ValueError("the query contains a write or schema-changing keyword")
    return cleaned


def run_query(query: str, database: str | None = None, limit: int = MAX_ROWS) -> dict[str, Any]:
    path = resolve_database(database)
    statement = validate_query(query)
    limit = max(1, min(int(limit), MAX_ROWS))

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        cursor = connection.execute(statement)
        rows = cursor.fetchmany(limit)
        columns = [description[0] for description in (cursor.description or [])]
        return {
            "ok": True,
            "columns": columns,
            "row_count": len(rows),
            "truncated": len(rows) >= limit,
            "rows": [dict(row) for row in rows],
        }
    finally:
        connection.close()


@tool(name="sql_query", tags=("data",))
def sql_query(query: str, limit: int = 50) -> dict[str, Any]:
    """Run a read-only SELECT against the configured SQLite database.

    Args:
        query: A single SELECT or WITH statement. Writes are rejected.
        limit: Maximum number of rows to return.
    """
    try:
        return run_query(query, limit=limit)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@tool(name="sql_schema", tags=("data",))
def sql_schema() -> dict[str, Any]:
    """List tables and columns in the configured SQLite database."""
    try:
        path = resolve_database()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        tables = [
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        schema = {}
        for table in tables:
            columns = connection.execute(f"PRAGMA table_info('{table}')").fetchall()
            schema[table] = [{"name": c["name"], "type": c["type"]} for c in columns]
        return {"ok": True, "tables": tables, "schema": schema}
    finally:
        connection.close()
