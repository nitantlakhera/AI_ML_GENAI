# Troubleshooting

Common issues and how to fix them.

---

## Setup issues

### `uv` is not recognized

**Cause:** uv not installed or not in PATH.

**Fix:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
```

### `llama-cpp-python` build fails on Windows

**Cause:** Missing C++ compiler (nmake, MSVC).

**Fix:** Use API LLM instead:

```env
USE_API_LLM=true
OPENAI_API_KEY=sk-your-key
```

Or install Visual Studio Build Tools and run:

```powershell
uv sync --extra local-llm
```

---

## RAG issues

### `No documents found in data/raw`

**Cause:** No supported files in `data/raw/`.

**Fix:** Add `.pdf`, `.txt`, or `.md` files, then re-run:

```powershell
uv run python ingest.py
```

### `Run ingest.py first to build the vector index`

**Cause:** `vector_db/` does not exist or is empty.

**Fix:**

```powershell
uv run python ingest.py
```

### Poor RAG answers / wrong sources

**Possible causes and fixes:**

| Issue | Fix |
|-------|-----|
| Chunks too large | Lower `CHUNK_SIZE` in `.env`, re-ingest |
| Too little context | Increase `TOP_K` |
| Wrong embedding model | Change `EMBEDDING_MODEL`, re-ingest |
| Stale index | Re-run `ingest.py` after adding documents |

### Slow first run / embedding download

**Cause:** HuggingFace downloads embedding model on first use.

**Fix:** Wait for download to complete. Model is cached for future runs.

---

## Application issues

### Streamlit shows blank or errors on load

**Fix:**

```powershell
uv run streamlit run app.py --server.headless true
```

Check terminal for Python tracebacks.

### Agent mode errors

**Cause:** Local GGUF models often lack tool-calling support.

**Fix:** Use API LLM:

```env
USE_API_LLM=true
OPENAI_API_KEY=sk-your-key
```

### `ImportError: llama-cpp-python is not installed`

**Cause:** `USE_API_LLM=false` but local-llm extra not installed.

**Fix:** Either:

```powershell
uv sync --extra local-llm
```

Or set `USE_API_LLM=true` in `.env`.

---

## MCP issues

### MCP server does not start in Cursor

**Checks:**

1. `cwd` in config points to project root (absolute path)
2. uv is in PATH for Cursor's environment
3. Use full path to uv if needed:

```json
{
  "command": "C:\\Users\\YOUR_USER\\.local\\bin\\uv.exe",
  "args": ["run", "python", "mcp_server/server.py"],
  "cwd": "C:\\MY_SPACE\\AI_ML_GENAI"
}
```

### Tools not appearing in client

**Fix:** Restart MCP client after changing `mcp_server/server.py`. Verify server runs manually:

```powershell
uv run python mcp_server/server.py
```

---

## Testing issues

### Tests fail on import

**Fix:** Run from project root:

```powershell
cd C:\MY_SPACE\AI_ML_GENAI
uv run pytest
```

### pytest slow on first run

**Cause:** torch/sentence-transformers loading.

**Fix:** Normal on first run. Subsequent runs are faster.

---

## Getting help

1. Check [Flow Diagrams](flow-diagrams.md) for expected behavior
2. Review [Configuration](configuration.md) for setting mistakes
3. Run tests: `uv run pytest -v`
4. Check logs in terminal when running `app.py` or `ingest.py`
