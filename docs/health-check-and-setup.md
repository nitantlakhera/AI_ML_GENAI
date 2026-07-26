# Health Check & Setup Status

Last verified: **July 2026** on the development laptop (Windows 11, Intel Core Ultra 5 135H, 32 GB RAM).

Use this page to confirm the project is working and to finish setup for chat, RAG answers, and agents.

---

## Quick status

| Area | Status |
|------|--------|
| Unit tests (`uv run pytest`) | **5/5 passed** |
| RAG ingest (`uv run python ingest.py`) | **Working** |
| Embeddings (MiniLM) | **Working** |
| FAISS vector DB | **Working** |
| MCP server (`mcp_server/`) | **Working** |
| Agent tools (calculator, word_count) | **Working** |
| Documentation + PNG diagrams | **Complete** |
| LLM (chat / answers) | **Needs configuration** — see below |

---

## What is working

| Component | Details |
|-----------|---------|
| **Unit tests** | All 5 tests pass |
| **Module imports** | `rag/`, `agents/`, `chat/`, `mcp_server/`, `config/` |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim vectors) |
| **RAG ingest** | Builds FAISS index from `data/raw/` |
| **FAISS vector DB** | `vector_db/` loads correctly |
| **MCP server** | Tools: `greet`, `word_count` |
| **Sample data** | `data/raw/sample.txt` included for testing |
| **`.env`** | Copy from `.env.example` if missing |

### Verify yourself

```powershell
cd C:\MY_SPACE\AI_ML_GENAI
uv run pytest -v
uv run python ingest.py
uv run python -c "from mcp_server.server import mcp; print(list(mcp._tool_manager._tools.keys()))"
```

Expected: tests pass, ingest indexes chunks, MCP prints `['greet', 'word_count']`.

---

## What you still need to configure

The LLM is **not ready** until you choose one option below. Without this, **Chatbot**, **AI Assistant**, **RAG Q&A**, and **Agent** modes cannot generate answers.

| Item | Current state | Action |
|------|---------------|--------|
| **LLM** | No GGUF in `models/`, `USE_API_LLM=false` | Pick Option A or B below |
| **llama-cpp-python** | Not installed by default | Only needed for local GGUF (Option B) |

---

## Option A — OpenAI API (fastest to test)

Best for quick testing and **Agent mode**.

1. Edit `.env`:

```env
USE_API_LLM=true
OPENAI_API_KEY=sk-your-key-here
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_TEMPERATURE=0.2
```

2. Run the app:

```powershell
uv run streamlit run app.py
```

No local model download required.

---

## Option B — Local GGUF (this laptop)

Best for offline use. On this laptop, prefer **3B Q4_K_M** for speed. See [Models — this laptop profile](models.md#this-laptop--system-profile--recommended-models).

1. Install local LLM support:

```powershell
uv sync --extra local-llm
```

2. Download a model (example — fast 3B):

```powershell
uv add huggingface-hub
uv run huggingface-cli download TheBloke/Llama-3.2-3B-Instruct-GGUF `
  Llama-3.2-3B-Instruct-Q4_K_M.gguf `
  --local-dir models
```

3. Edit `.env`:

```env
USE_API_LLM=false
LLM_MODEL_PATH=models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
LLM_N_CTX=4096
LLM_TEMPERATURE=0.2
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

4. Build index (if not done) and run:

```powershell
uv run python ingest.py
uv run streamlit run app.py
```

---

## How to run now

Always from project root:

```powershell
cd C:\MY_SPACE\AI_ML_GENAI
```

| Task | Command |
|------|---------|
| **Main app (all modes)** | `uv run streamlit run app.py` |
| **REST API + Swagger** | `uv run python api_server.py` → http://localhost:8080/docs |
| **Build / update RAG index** | `uv run python ingest.py` |
| **MCP server** | `uv run python mcp_server/server.py` |
| **Run tests** | `uv run pytest` |
| **Activate venv (optional)** | `.\.venv\Scripts\Activate.ps1` |

Open the Streamlit URL in your browser (usually **http://localhost:8501**).

### After LLM is configured

| Mode | Expected behavior |
|------|-------------------|
| **RAG Q&A** | Answers from `data/raw/` (sample doc works out of the box) |
| **Chatbot** | Conversational replies |
| **AI Assistant** | Replies with optional RAG |
| **Agent** | Tool-calling (best with OpenAI API) |

---

## Fixes applied during health check

These issues were found and **fixed in the codebase**:

| Issue | Fix |
|-------|-----|
| LangChain 1.x import errors (`langchain.chains`, `langchain.agents`) | Switched to `langchain_classic` imports |
| Local `mcp/` folder shadowed PyPI `mcp` package | Renamed to `mcp_server/` |
| No sample documents for RAG testing | Added `data/raw/sample.txt` |
| Missing `.env` | Use `copy .env.example .env` |

**MCP config path:** `mcp_server/config.json`  
**MCP server entry:** `uv run python mcp_server/server.py`

---

## Minor warnings (non-blocking)

| Warning | Impact | Optional fix |
|---------|--------|--------------|
| `HuggingFaceEmbeddings` deprecation | Still works | Migrate to `langchain-huggingface` later |
| HuggingFace unauthenticated requests | Slower first download | Set `HF_TOKEN` in `.env` |
| FAISS AVX2 not loaded | Uses standard FAISS | No action needed on most PCs |

---

## Full verification checklist

```powershell
cd C:\MY_SPACE\AI_ML_GENAI

# 1. Dependencies
uv sync --extra dev

# 2. Environment
copy .env.example .env
# Edit .env — add API key OR local model path

# 3. Tests
uv run pytest -v

# 4. RAG index
uv run python ingest.py

# 5. MCP server (optional)
uv run python mcp_server/server.py

# 6. App
uv run streamlit run app.py
```

---

## Related docs

- [Run, Activate & Dependencies](run-and-dependencies.md)
- [Models Guide](models.md)
- [This laptop profile](models.md#this-laptop--system-profile--recommended-models)
- [Getting Started](getting-started.md)
- [Troubleshooting](troubleshooting.md)
