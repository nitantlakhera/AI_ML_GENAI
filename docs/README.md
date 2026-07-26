# Documentation

Welcome to the **AI / ML / GenAI** project documentation. This folder contains architecture diagrams, user manuals, flow diagrams, and configuration references.

## Documentation index

| Document | Description |
|----------|-------------|
| [Getting Started](getting-started.md) | First-time setup and quick start |
| [Run, Activate & Dependencies](run-and-dependencies.md) | How to run, activate venv, add packages with uv |
| [MiniGPT (merged)](minigpt.md) | Tiny GPT + Llama fine-tuning in `minigpt/` |
| [REST API (Swagger)](api.md) | FastAPI endpoints and `/docs` |
| [Health Check & Setup Status](health-check-and-setup.md) | What's working, LLM setup, verification checklist |
| [Models Guide](models.md) | How to use models, where to get them, all usage modes |
| [Models — this laptop profile](models.md#this-laptop--system-profile--recommended-models) | Hardware profile and model picks for this machine |
| [User Manual](user-manual.md) | How to use every feature (chatbot, assistant, RAG, agents, MCP) |
| [Architecture](architecture.md) | System design, modules, and data flow |
| [Flow Diagrams](flow-diagrams.md) | Step-by-step flows with Mermaid + PNG diagrams |
| [Configuration Reference](configuration.md) | Environment variables and settings |
| [Troubleshooting](troubleshooting.md) | Common issues and fixes |

## Architecture diagrams (PNG)

| Diagram | File | Description |
|---------|------|-------------|
| System overview | [images/architecture-overview.png](images/architecture-overview.png) | High-level architecture |
| RAG pipeline | [images/rag-flow.png](images/rag-flow.png) | Document ingestion and Q&A flow |
| Agent flow | [images/agent-flow.png](images/agent-flow.png) | Tool-calling agent loop |
| MCP flow | [images/mcp-flow.png](images/mcp-flow.png) | MCP server/client integration |
| App modes | [images/app-modes-flow.png](images/app-modes-flow.png) | Streamlit UI mode routing |

## PDF documentation (`docs/pdf/`)

Downloadable PDFs for offline reading and sharing:

| PDF | File | Contents |
|-----|------|----------|
| **User Manual** | [pdf/user-manual.pdf](pdf/user-manual.pdf) | Setup, run commands, all app modes, API, MiniGPT |
| **Architecture** | [pdf/architecture.pdf](pdf/architecture.pdf) | System design + architecture diagram |
| **Flow Diagrams** | [pdf/flow-diagrams.pdf](pdf/flow-diagrams.pdf) | All flow PNG diagrams + flow descriptions |
| **Complete Guide** | [pdf/complete-guide.pdf](pdf/complete-guide.pdf) | Combined manual, architecture, models, diagrams |

Regenerate PDFs after doc updates:

```powershell
uv run python scripts/generate_pdfs.py
```

## Quick links

See **[Run, Activate & Dependencies](run-and-dependencies.md)** for full details on venv activation and adding packages.

```powershell
# First-time setup
uv sync --extra dev
copy .env.example .env

# Activate venv (optional — uv run works without this)
.\.venv\Scripts\Activate.ps1

# Build RAG index
uv run python ingest.py

# Run app
uv run streamlit run app.py

# Run API (Swagger at http://localhost:8080/docs)
uv run python api_server.py

# Run MCP server
uv run python mcp_server/server.py

# Run tests
uv run pytest

# Add a new dependency
uv add <package-name>

# Download a GGUF model (example)
uv add huggingface-hub
uv run huggingface-cli download TheBloke/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir models
```

## Project modules

```
config/     → Settings and environment
rag/        → Retrieval-Augmented Generation
agents/     → AI agents with tools
mcp_server/        → Model Context Protocol server/client
minigpt/    → MiniGPT (PyTorch, TensorFlow, Llama fine-tune)
app.py      → Main Streamlit application
ingest.py   → Vector database builder
```
