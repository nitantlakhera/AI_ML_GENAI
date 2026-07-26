# Configuration Reference

All settings are defined in `config/settings.py` and loaded from `.env`.

Copy the template:

```powershell
copy .env.example .env
```

---

## Environment variables

### Embeddings

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model for vector embeddings |

### Local LLM (GGUF)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL_PATH` | `models/llama-3-8b-instruct.Q4_K_M.gguf` | Path to GGUF model file |
| `LLM_N_CTX` | `4096` | Context window size |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature (0 = deterministic) |

**Install local LLM support:**

```powershell
uv sync --extra local-llm
```

### API LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_API_LLM` | `false` | Set `true` to use OpenAI instead of local GGUF |
| `OPENAI_API_KEY` | (empty) | OpenAI API key when `USE_API_LLM=true` |

### RAG

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `500` | Characters per text chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `TOP_K` | `4` | Number of chunks retrieved per query |

**Tuning tips:**

- Larger `CHUNK_SIZE` → more context per chunk, fewer chunks
- Higher `TOP_K` → more context for LLM, slower inference
- Re-run `ingest.py` after changing chunk settings

### Agents

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MAX_ITERATIONS` | `10` | Max tool-calling loops per agent run |

### MCP

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SERVER_HOST` | `localhost` | MCP server host (for future HTTP transport) |
| `MCP_SERVER_PORT` | `8000` | MCP server port |

---

## Path constants (settings.py)

These are computed in code — do not set in `.env`:

| Constant | Path | Description |
|----------|------|-------------|
| `BASE_DIR` | Project root | Auto-detected |
| `DATA_RAW_DIR` | `data/raw/` | Source documents |
| `DATA_PROCESSED_DIR` | `data/processed/` | Processed exports |
| `VECTOR_DB_DIR` | `vector_db/` | FAISS index |
| `MODELS_DIR` | `models/` | GGUF weights |
| `MCP_CONFIG_PATH` | `mcp_server/config.json` | MCP client config |

---

## Example configurations

### API-only setup (recommended for agents)

```env
USE_API_LLM=true
OPENAI_API_KEY=sk-your-key
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=500
TOP_K=4
```

### Local LLM setup

```env
USE_API_LLM=false
LLM_MODEL_PATH=models/llama-3-8b-instruct.Q4_K_M.gguf
LLM_N_CTX=4096
LLM_TEMPERATURE=0.2
```

### High-recall RAG

```env
CHUNK_SIZE=300
CHUNK_OVERLAP=80
TOP_K=8
```

---

## pyproject.toml extras

| Extra | Command | Includes |
|-------|---------|----------|
| `dev` | `uv sync --extra dev` | pytest, jupyter, ipykernel |
| `local-llm` | `uv sync --extra local-llm` | llama-cpp-python for GGUF models |

See [Models Guide](models.md) for where to download GGUF files and how to configure them.
