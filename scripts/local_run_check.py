"""Local readiness check — run: uv run python scripts/local_run_check.py"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ok, warn, fail = [], [], []

    try:
        from config.settings import (
            DATA_RAW_DIR,
            LLM_MODEL_PATH,
            OPENAI_API_KEY,
            USE_API_LLM,
            VECTOR_DB_DIR,
        )
        from rag.loader import load_documents
        from rag.embeddings import get_embeddings
        from rag.vector_store import load_vector_store
        from rag.chain import build_rag_chain
        from chat.bot import ChatBot
        from chat.assistant import AIAssistant
        from agents.base import build_agent_executor
        from agents.tools import calculator
        from mcp_server.server import mcp
        from api.main import app

        ok.append("All Python modules import")
    except Exception as exc:
        fail.append(f"Import error: {exc}")
        _print(ok, warn, fail)
        return 1

    try:
        emb = get_embeddings()
        vec = emb.embed_query("test")
        ok.append(f"Embeddings work (dim={len(vec)})")
    except Exception as exc:
        fail.append(f"Embeddings: {exc}")

    try:
        if VECTOR_DB_DIR.exists():
            load_vector_store(VECTOR_DB_DIR, emb)
            ok.append("FAISS vector_db loads")
        else:
            warn.append("vector_db missing — run: uv run python ingest.py")
    except Exception as exc:
        fail.append(f"FAISS: {exc}")

    try:
        tools = list(mcp._tool_manager._tools.keys())
        ok.append(f"MCP tools: {tools}")
    except Exception as exc:
        fail.append(f"MCP: {exc}")

    if USE_API_LLM:
        if OPENAI_API_KEY:
            ok.append("LLM: OpenAI API key set")
        else:
            warn.append("USE_API_LLM=true but OPENAI_API_KEY is empty")
    elif Path(LLM_MODEL_PATH).exists():
        ok.append("LLM: local GGUF file found")
    else:
        warn.append("LLM not ready — set API key or download GGUF model")

    try:
        import llama_cpp  # noqa: F401

        ok.append("llama-cpp-python installed")
    except ImportError:
        warn.append("llama-cpp-python not installed (needed for local GGUF)")

    if Path(".env").exists():
        ok.append(".env exists")
    else:
        warn.append(".env missing — copy .env.example .env")

    docs = load_documents(DATA_RAW_DIR)
    ok.append(f"data/raw: {len(docs)} document page(s)")

    spec = importlib.util.spec_from_file_location(
        "minigpt_paths", ROOT / "minigpt/pytorch/paths.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ok.append(f"MiniGPT paths OK ({mod.COMMON.name})")

    pdfs = list((ROOT / "docs/pdf").glob("*.pdf"))
    ok.append(f"PDF docs: {len(pdfs)} file(s)")

    # API smoke test
    try:
        from fastapi.testclient import TestClient

        client = TestClient(app)
        r = client.get("/health")
        if r.status_code == 200:
            ok.append(f"API /health: {r.json()}")
        else:
            fail.append(f"API /health returned {r.status_code}")
    except Exception as exc:
        fail.append(f"API test: {exc}")

    # Streamlit app import (don't start server)
    try:
        import importlib

        importlib.import_module("app")
        ok.append("Streamlit app.py imports")
    except Exception as exc:
        fail.append(f"app.py import: {exc}")

    _print(ok, warn, fail)
    return 1 if fail else 0


def _print(ok, warn, fail):
    print("=== LOCAL RUN CHECK ===")
    for x in ok:
        print("[OK]", x)
    for x in warn:
        print("[WARN]", x)
    for x in fail:
        print("[FAIL]", x)
    print(f"Summary: {len(ok)} ok, {len(warn)} warnings, {len(fail)} failures")


if __name__ == "__main__":
    sys.exit(main())
