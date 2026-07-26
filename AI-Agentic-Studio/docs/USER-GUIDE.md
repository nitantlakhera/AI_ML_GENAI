# User Guide

How to use **AI-Agentic-Studio** v1 — CLI, API, UI, agents, RAG, evaluation, and MCP.

> **New to Generative or Agentic AI?** Read **[LEARNING-GUIDE.md](LEARNING-GUIDE.md)** first for concepts, install steps, call-flow diagrams, and a beginner learning path. This guide is the detailed reference once you are up and running.

All examples assume you are in the project root with the virtual environment activated:

```bash
cd AI-Agentic-Studio
.venv\Scripts\activate    # Windows
```

---

## 1. First-time setup

### Install

```bash
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[retrieval]"   # sentence-transformers + FAISS
pip install -e ".[ui]"          # Streamlit playground
pip install -e ".[mcp]"         # MCP server/client
```

### Configure (optional)

```bash
cp .env.example .env
```

Defaults work offline. Edit `.env` only when you want hosted models, Tavily search, or API auth.

### Verify

```bash
studio doctor
```

You should see `echo` as the active provider, the embedder name, corpus stats, and the tool list.

---

## 2. Document ingestion

### Index sample handbooks

```bash
studio ingest data/raw
```

### Index a specific file or folder

```bash
studio ingest path/to/my-docs/
studio ingest path/to/report.pdf
```

### What happens

1. Files are loaded (text, PDF, JSON, CSV, HTML).
2. Content is chunked (recursive by default; configurable via `STUDIO_CHUNK_STRATEGY`).
3. Chunks are embedded and stored in `var/index/`.
4. BM25 and knowledge-graph indexes are rebuilt.

### Via API

```bash
curl -X POST http://localhost:8100/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "data/raw"}'
```

Inline texts (no files):

```bash
curl -X POST http://localhost:8100/ingest \
  -H "Content-Type: application/json" \
  -d '{"texts": ["BM25 catches rare identifiers."], "source": "notes"}'
```

---

## 3. Grounded Q&A (RAG)

### CLI

```bash
# Full answer with citations
studio ask "What problem does reciprocal rank fusion solve?"

# JSON output
studio ask "Why combine BM25 with dense retrieval?" --json

# Retrieval only (no generation)
studio search "cross-encoder reranker"
```

### API

```bash
curl -X POST http://localhost:8100/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does BM25 catch that dense retrieval misses?"}'
```

Response fields: `answer`, `sources[]`, `queries_used`, `usage`, `latency_ms`.

### Streamed answer (SSE)

```bash
curl -N -X POST http://localhost:8100/rag/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain hybrid retrieval."}'
```

Events arrive as `data: {...}` lines — sources first, then answer tokens.

---

## 4. Conversational chat

Multi-turn chat with persistent memory and optional RAG grounding.

### API

```bash
# Start a thread
curl -X POST http://localhost:8100/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is reciprocal rank fusion?"}'

# Continue (use thread_id from response)
curl -X POST http://localhost:8100/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "thread_abc123", "message": "How does it compare to weighted fusion?"}'
```

### List and inspect threads

```bash
curl http://localhost:8100/threads
curl http://localhost:8100/threads/thread_abc123
```

### Streamlit UI

```bash
studio ui
```

Open the **Chat** tab. Each session gets a `thread_id`; history persists in `var/memory.sqlite3`.

---

## 5. Agents

Three modes are implemented:

| Mode | CLI flag | Best for |
|------|----------|----------|
| ReAct | `--mode react` (default) | General tool-calling tasks |
| Plan-execute | `--mode plan` | Multi-step research with critique |
| Supervisor team | `--mode team` | Tasks needing research + compute specialists |

### CLI examples

```bash
# ReAct: search corpus and summarise
studio agent "Use rag_search to find passages about BM25 and explain why it helps hybrid retrieval."

# Plan mode
studio agent "Compare BM25 and dense retrieval, then list when to use each." --mode plan

# Team mode
studio agent "Research RRF in the corpus and calculate 1234 * 17." --mode team

# JSON output (full run trace)
studio agent "What is human-in-the-loop?" --json
```

### API

```bash
curl -X POST http://localhost:8100/agent \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Search the corpus for cross-encoder reranking and summarise.",
    "mode": "react",
    "max_steps": 8
  }'
```

Restrict tools:

```json
{
  "task": "Calculate (999 * 888) / 2",
  "mode": "react",
  "tools": ["calculator"]
}
```

### Human-in-the-loop (HITL)

Tools marked `requires_approval` (`python_exec`, `write_file`, `delete_file`) pause the run.

1. Agent returns `status: "interrupted"` with `pending_approval`.
2. Approve or reject:

```bash
curl -X POST http://localhost:8100/agent/approvals/REQ_ID \
  -H "Content-Type: application/json" \
  -d '{"approved": true}'
```

3. The run resumes and returns the final output.

In the Streamlit UI, pending approvals appear in the sidebar and the **Agents** tab.

### View agent graph

```bash
studio graph
```

Prints Mermaid diagram of the ReAct think/act loop.

---

## 6. Tools reference

List all tools:

```bash
studio tools
```

| Tool | Description | Approval |
|------|-------------|----------|
| `calculator` | Safe arithmetic expressions | No |
| `web_search` | Web or offline corpus search | No |
| `rag_search` | Retrieve passages from index | No |
| `rag_answer` | Full grounded answer with citations | No |
| `graph_explore` | Knowledge-graph entity lookup | No |
| `corpus_stats` | Index statistics | No |
| `list_sources` | Distinct document sources | No |
| `list_files` | List sandbox files | No |
| `read_file` | Read sandbox file | No |
| `write_file` | Write sandbox file | **Yes** |
| `delete_file` | Delete sandbox file | **Yes** |
| `sql_query` | Read-only SQLite SELECT | No |
| `sql_schema` | List tables and columns | No |
| `http_request` | Allowlisted HTTP GET/POST | No |
| `python_exec` | Sandboxed Python execution | **Yes** |

### Configure HTTP allowlist

```env
STUDIO_HTTP_ALLOWED_HOSTS=api.github.com,jsonplaceholder.typicode.com
```

### Configure SQL database

```env
STUDIO_SQL_DATABASE_URL=sqlite:///path/to/data.db
```

### Configure web search

```env
STUDIO_SEARCH_PROVIDER=offline    # default; searches local corpus
STUDIO_SEARCH_PROVIDER=tavily     # requires TAVILY_API_KEY
STUDIO_SEARCH_PROVIDER=duckduckgo
```

---

## 7. Evaluation

Score the RAG pipeline against a golden question set.

### Create and run

```bash
studio eval
```

On first run, a starter golden set is created at `data/eval/golden.jsonl`.

### Compare advanced vs naive pipeline

```bash
studio eval --compare
```

### Use LLM-as-judge (requires hosted provider)

```bash
STUDIO_LLM_PROVIDERS=openai
OPENAI_API_KEY=sk-...
studio eval --judge
```

Reports are written to `reports/` as JSON and Markdown.

### Via API (background job)

```bash
curl -X POST http://localhost:8100/eval/run \
  -H "Content-Type: application/json" \
  -d '{"label": "smoke", "dataset": "data/eval/golden.jsonl"}'

curl http://localhost:8100/jobs/JOB_ID
```

---

## 8. REST API

### Start the server

```bash
studio serve
# or with reload:
studio serve --reload
```

Swagger UI: **http://localhost:8100/docs**

### Enable API key auth

```env
STUDIO_API_KEYS=my-secret-key
```

Then pass the key:

```bash
curl -H "X-API-Key: my-secret-key" http://localhost:8100/health
```

### Observability endpoints

```bash
curl http://localhost:8100/metrics    # counters and histograms
curl http://localhost:8100/traces     # recent trace spans
```

---

## 9. Streamlit UI

```bash
pip install -e ".[ui]"
studio ui
```

Opens **http://localhost:8501** with six tabs:

| Tab | Purpose |
|-----|---------|
| **Chat** | Streaming conversational RAG with sources |
| **Retrieval** | Inspect ranked passages without generation |
| **Agents** | Run agents, view steps, approve tool calls |
| **Ingest** | Upload and index documents |
| **Eval** | Run evaluation and view aggregate scores |
| **Observability** | Metrics snapshot and recent traces |

Sidebar shows active providers, corpus stats, retrieval settings, and pending approvals.

---

## 10. MCP integration

### Expose studio tools to Cursor / Claude Desktop

```bash
pip install -e ".[mcp]"
studio mcp-serve
```

Publishes all **safe** tools (no `python_exec`, `write_file`, `delete_file`).

Example Cursor MCP config (`agentic_studio/mcp_bridge/config.json`):

```json
{
  "mcpServers": {
    "ai-agentic-studio": {
      "command": "python",
      "args": ["-m", "agentic_studio.mcp_bridge.server"]
    }
  }
}
```

### Bridge external MCP tools into agents

```bash
studio mcp-register --config path/to/mcp-servers.json
```

External tools are added to `REGISTRY` and become available to agents.

---

## 11. Switching to hosted / local models

### OpenAI

```env
STUDIO_LLM_PROVIDERS=openai,echo
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

### Anthropic

```env
STUDIO_LLM_PROVIDERS=anthropic,echo
ANTHROPIC_API_KEY=sk-ant-...
```

### Ollama (local)

```env
STUDIO_LLM_PROVIDERS=ollama,echo
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
```

### Semantic embeddings

```bash
pip install -e ".[retrieval]"
```

```env
STUDIO_EMBEDDING_BACKEND=sentence-transformers
STUDIO_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Re-ingest after changing embedder:

```bash
studio ingest data/raw
```

---

## 12. Guardrails

Enabled by default (`STUDIO_GUARDRAILS_ENABLED=true`).

| Setting | Options | Default |
|---------|---------|---------|
| `STUDIO_PII_MODE` | `redact`, `block`, `off` | `redact` |
| `STUDIO_MODERATION_MODE` | `block`, `warn` | `block` |
| `STUDIO_MAX_INPUT_CHARS` | integer | `20000` |

Blocked inputs return HTTP 422 with a `rule` field (e.g. `moderation`, `pii`).

Retrieved documents are sanitized before entering prompts to reduce indirect prompt injection.

---

## 13. Troubleshooting

| Problem | Solution |
|---------|----------|
| Empty answers / no sources | Run `studio ingest data/raw` first |
| `AllProvidersFailed` | Check API keys; ensure `echo` is last in provider chain as fallback |
| Agent pauses mid-run | Approve the tool call via API or Streamlit UI |
| `faiss not installed` | Normal — numpy search is used; install `[retrieval]` for FAISS |
| `mcp package not installed` | `pip install -e ".[mcp]"` |
| Streamlit won't start | `pip install -e ".[ui]"` |
| SQL tool errors | Set `STUDIO_SQL_DATABASE_URL` to a valid SQLite file |

### Reset local state

```bash
rm -rf var/
studio ingest data/raw
```

---

## 14. Python API (programmatic use)

```python
from agentic_studio.rag.pipeline import get_pipeline
from agentic_studio.agents.react import ToolCallingAgent

# RAG
result = get_pipeline().answer("What is hybrid retrieval?")
print(result.answer)

# Agent
run = ToolCallingAgent().run("Search the corpus for BM25 and summarise.")
print(run.output)
```

All modules are importable from the `agentic_studio` package. Settings are read from the environment on first access.
