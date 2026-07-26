# AI Agentic Studio

A **production-shaped Generative + Agentic AI playground** built as a separate project inside `AI_ML_GENAI`. It extends the original repo with advanced RAG, stateful agents, guardrails, observability, evaluation, and multiple interfaces (CLI, REST API, Streamlit UI, MCP).

Runs **offline by default** — no API keys required. The `echo` LLM provider and `hashing` embedder make demos and tests deterministic.

> **Just want to run it?** Open **[START-HERE.md](START-HERE.md)** — one page: install, commands, UI, API.  
> **Follow a 4-week plan?** **[docs/LEARNING-PATH.md](docs/LEARNING-PATH.md)** — day-by-day beginner journey with hands-on labs.  
> **Learn every concept?** **[docs/CONCEPT-CAPSULES.md](docs/CONCEPT-CAPSULES.md)** — 66 bite-sized capsules · **[docs/CONCEPTS.md](docs/CONCEPTS.md)** — reference tables

## What is implemented

| Area | Status |
|------|--------|
| Multi-provider LLM router (failover, cache, streaming, structured output) | Done |
| Advanced RAG (hybrid BM25+dense, RRF, rerank, query transforms, graph RAG) | Done |
| Stateful agents (ReAct, plan-execute, supervisor) with HITL | Done |
| Tool suite (search, RAG-as-tool, filesystem, SQL, HTTP, sandboxed Python) | Done |
| Persistent memory (SQLite) + summarizing window | Done |
| Guardrails (PII, moderation, injection, tool policy) | Done |
| Observability (structured logs, metrics, tracing, optional LangSmith/OTel) | Done |
| Evaluation harness (lexical metrics, LLM-as-judge, golden datasets) | Done |
| FastAPI + SSE streaming + background jobs | Done |
| Streamlit playground UI | Done |
| MCP server (expose tools) + MCP client (bridge external tools) | Done |
| Multimodal helpers (image data URLs, vision model wiring) | Done |
| Offline pytest suite | Done |

See [docs/IMPLEMENTATION-STATUS.md](docs/IMPLEMENTATION-STATUS.md) for the full module-by-module checklist.

## Coverage summary

All **generative and agentic AI gaps** identified for the v1 playground scope are closed in this project. Details: [docs/GAP-ANALYSIS.md](docs/GAP-ANALYSIS.md).

### Covered (22/22)

| Category | Gaps closed |
|----------|-------------|
| **Generative / RAG** | Hybrid BM25+dense, RRF fusion, reranking, query transforms, graph RAG, parent-doc context, metadata filters, conversational RAG |
| **LLM layer** | Multi-provider router, failover, caching, structured output |
| **Agentic** | StateGraph engine, checkpointing, ReAct + plan-execute + supervisor, HITL approvals, 14+ real tools with safety |
| **Platform** | Guardrails, observability, evaluation harness, FastAPI, CLI, Streamlit UI, MCP bridge |
| **Engineering** | Offline defaults (`echo` + `hashing`), central config, pytest suite, documentation |

### Caveats (implemented, with limits)

| Feature | What you get | Limitation |
|---------|--------------|------------|
| Graph RAG | Entity co-occurrence graph from chunks | Not a full knowledge-graph DB (Neo4j, etc.) |
| Multimodal | Image helpers + vision message wiring | No dedicated vision chat tab or end-to-end vision demo |
| MCP | Server + client bridge | Requires `pip install -e ".[mcp]"` |
| Semantic embeddings / FAISS | Optional extras | Default offline mode uses hashing + numpy |
| Cross-encoder rerank | Supported when installed | Default reranker is lexical |
| LLM-as-judge eval | Supported | Needs a hosted provider key |
| Python sandbox | Subprocess + import blocklist | Not container/gVisor isolation |
| Agent streaming | SSE for ReAct | Plan/team modes do not stream yet |

### Out of scope (by design)

These were not part of the v1 gap list — they remain in the parent repo or [docs/ROADMAP.md](docs/ROADMAP.md):

- MiniGPT / fine-tuning (`AI_ML_GENAI/MiniGPT/`)
- Docker / Kubernetes deployment manifests
- Cloud vector databases (Pinecone, Weaviate, pgvector)
- Voice / video agents
- Multi-tenant RBAC beyond API keys
- Distributed agent workers

## Quick start

```bash
cd AI-Agentic-Studio
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -e ".[dev]"
cp .env.example .env              # optional; defaults work offline

# Check what is available
studio doctor

# Index sample handbooks
studio ingest data/raw

# Ask a grounded question
studio ask "What does BM25 catch that dense retrieval misses?"

# Run a tool-calling agent
studio agent "Search the corpus for reciprocal rank fusion and summarise it."

# Start the REST API (Swagger at http://localhost:8100/docs)
studio serve

# Start the Streamlit UI
pip install -e ".[ui]"
studio ui
```

## Project layout

```
AI-Agentic-Studio/
├── agentic_studio/          # Python package
│   ├── llm/                 # Provider router, cache, structured output
│   ├── rag/                 # Ingestion, retrieval, generation pipeline
│   ├── agents/              # StateGraph engine, ReAct/plan/team agents, tools
│   ├── memory/              # SQLite conversation store + summarizing memory
│   ├── guardrails/          # PII, moderation, policy
│   ├── observability/       # Logs, metrics, tracing
│   ├── evaluation/          # Metrics, judge, runner, datasets
│   ├── api/                   # FastAPI application
│   ├── ui/                  # Streamlit playground
│   ├── mcp_bridge/          # MCP server + client
│   └── cli.py               # `studio` command
├── data/raw/                # Sample documents (retrieval + agent handbooks)
├── data/eval/               # Golden evaluation sets (JSONL)
├── tests/                   # Offline pytest suite
├── docs/                    # Architecture, user guide, gap analysis
├── .env.example
└── pyproject.toml
```

## Documentation

| Document | Description |
|----------|-------------|
| **[START-HERE.md](START-HERE.md)** | **One page** — install, run, use (start here if you want commands only) |
| **[docs/LEARNING-PATH.md](docs/LEARNING-PATH.md)** | **4-week beginner path** — daily plan, labs, checkpoints, capstone |
| **[docs/CONCEPT-CAPSULES.md](docs/CONCEPT-CAPSULES.md)** | 66 concept capsules — one idea per card |
| **[docs/CONCEPTS.md](docs/CONCEPTS.md)** | Full concept reference tables (70+ terms) |
| **[docs/LEARNING-GUIDE.md](docs/LEARNING-GUIDE.md)** | Beginner path + labs |
| [docs/USER-GUIDE.md](docs/USER-GUIDE.md) | Step-by-step usage: CLI, API, UI, agents, MCP |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and architecture diagrams (Mermaid + PNG) |
| [docs/diagrams/](docs/diagrams/) | **PNG diagram images** — architecture + call flows (regenerate: `python docs/render_diagrams.py`) |
| [docs/IMPLEMENTATION-STATUS.md](docs/IMPLEMENTATION-STATUS.md) | What is built, file map, configuration reference |
| [docs/GAP-ANALYSIS.md](docs/GAP-ANALYSIS.md) | Gaps in the original repo and how this project closes them |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Future ideas (not yet implemented) |

## Configuration

All settings are environment variables. Copy `.env.example` to `.env` and edit as needed.

Key defaults (no keys required):

```env
STUDIO_LLM_PROVIDERS=echo
STUDIO_EMBEDDING_BACKEND=hashing
STUDIO_SEARCH_PROVIDER=offline
STUDIO_GUARDRAILS_ENABLED=true
```

To use hosted models, set provider keys and chain:

```env
STUDIO_LLM_PROVIDERS=openai,echo
OPENAI_API_KEY=sk-...
```

For semantic embeddings and FAISS acceleration:

```bash
pip install -e ".[retrieval]"
STUDIO_EMBEDDING_BACKEND=sentence-transformers
```

## CLI commands

| Command | Purpose |
|---------|---------|
| `studio doctor` | Check providers, corpus, tools, optional packages |
| `studio ingest [path]` | Index files from `data/raw` or a given path |
| `studio ask "..."` | Grounded answer with citations |
| `studio search "..."` | Retrieval only (inspect ranked passages) |
| `studio agent "..."` | Run agent (`--mode react\|plan\|team`) |
| `studio eval` | Score pipeline against golden set |
| `studio serve` | Start FastAPI on port 8100 |
| `studio ui` | Start Streamlit playground |
| `studio tools` | List registered tools |
| `studio graph` | Print agent graph as Mermaid |
| `studio mcp-serve` | Expose safe tools over MCP |
| `studio mcp-register --config FILE` | Bridge external MCP tools into agents |

## API overview

Interactive docs: **http://localhost:8100/docs**

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Health, providers, corpus stats |
| `POST /ingest` | Index documents (background job for directories) |
| `POST /rag/query` | Grounded answer with sources |
| `POST /rag/search` | Retrieval only |
| `POST /chat` | Multi-turn conversational RAG |
| `POST /agent` | Run agent (react / plan / team) |
| `POST /agent/approvals/{id}` | Approve or reject a paused tool call |
| `POST /eval/run` | Run evaluation as background job |
| `GET /metrics` | In-process counters and histograms |
| `GET /tools` | List all registered tools |

API key auth is optional. Set `STUDIO_API_KEYS=key1,key2` to enable.

## Tests

```bash
pytest tests -q
```

The suite runs fully offline using the `echo` provider and in-memory / tmp-path fixtures.

## Relationship to the parent repo

`AI-Agentic-Studio` lives alongside the original `AI_ML_GENAI` examples (basic RAG chatbots, LangChain demos, MiniGPT). It is a **separate, self-contained package** with its own `pyproject.toml`, focused on closing generative and agentic AI gaps documented in [docs/GAP-ANALYSIS.md](docs/GAP-ANALYSIS.md).

## License

Same as the parent `AI_ML_GENAI` repository.
