# How to Run, Activate, and Manage Dependencies

This guide covers daily development workflow: activating the environment, running the project, and adding new packages with **uv**.

---

## Table of contents

1. [Activate the virtual environment](#activate-the-virtual-environment)
2. [How to run the project](#how-to-run-the-project)
3. [Add new dependencies](#add-new-dependencies)
4. [Daily workflow cheat sheet](#daily-workflow-cheat-sheet)

---

## Activate the virtual environment

This project uses **uv** and a virtual environment at `.venv/` (created automatically on first `uv sync`).

You have two options — **Option A is recommended** because you do not need to activate manually.

### Option A: Use `uv run` (recommended — no activation needed)

`uv run` automatically uses the project `.venv` for any command:

```powershell
cd C:\MY_SPACE\AI_ML_GENAI

uv run python ingest.py
uv run streamlit run app.py
uv run pytest
```

No `activate` step required.

### Option B: Activate `.venv` manually

Use this if you prefer a traditional workflow or run many commands in the same shell session.

**Windows PowerShell:**

```powershell
cd C:\MY_SPACE\AI_ML_GENAI
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt (cmd):**

```cmd
cd C:\MY_SPACE\AI_ML_GENAI
.venv\Scripts\activate.bat
```

**Linux / macOS:**

```bash
cd /path/to/AI_ML_GENAI
source .venv/bin/activate
```

**Verify activation** — your prompt should show `(.venv)` and Python should point to the venv:

```powershell
python --version
where python        # Windows
which python        # Linux/macOS
```

**Deactivate** when done:

```powershell
deactivate
```

### If `.venv` does not exist yet

Create it and install dependencies first:

```powershell
cd C:\MY_SPACE\AI_ML_GENAI
uv sync --extra dev
```

---

## How to run the project

Always start from the project root:

```powershell
cd C:\MY_SPACE\AI_ML_GENAI
```

### First-time / full setup run

```powershell
# 1. Install dependencies (creates .venv)
uv sync --extra dev

# 2. Copy environment config
copy .env.example .env
# Edit .env — set OPENAI_API_KEY if using API LLM

# 3. Add documents to data/raw/ (PDF, TXT, MD)

# 4. Build vector index (required for RAG modes)
uv run python ingest.py

# 5. Start the web app
uv run streamlit run app.py
```

Open the browser URL from the terminal (usually **http://localhost:8501**).

### Run commands reference

| What to run | Command | Notes |
|-------------|---------|-------|
| **Main app (Streamlit UI)** | `uv run streamlit run app.py` | Chatbot, Assistant, RAG, Agent modes |
| **REST API + Swagger** | `uv run python api_server.py` | Open **http://localhost:8080/docs** |
| **Build / update RAG index** | `uv run python ingest.py` | Run after adding documents |
| **MCP server** | `uv run python mcp_server/server.py` | For Cursor / MCP clients |
| **Run tests** | `uv run pytest` | All unit tests |
| **Run tests (verbose)** | `uv run pytest -v` | Detailed output |
| **Jupyter notebooks** | `uv run jupyter lab notebooks/` | Experiments notebook |
| **Single Python script** | `uv run python your_script.py` | Any script in the project |

### Run with activated venv (Option B)

If you activated `.venv` manually, drop the `uv run` prefix:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
python ingest.py
pytest
jupyter lab notebooks/
```

### Optional: local GGUF model support

```powershell
uv sync --extra local-llm
```

Then place a `.gguf` file in `models/` and set `LLM_MODEL_PATH` in `.env`.

### Stop the application

- **Streamlit / Jupyter:** Press `Ctrl + C` in the terminal
- **MCP server:** Press `Ctrl + C` in the terminal

---

## Add new dependencies

Dependencies are managed in **`pyproject.toml`** using **uv** (not `pip install` directly).

### Add a runtime dependency

```powershell
cd C:\MY_SPACE\AI_ML_GENAI
uv add <package-name>
```

**Examples:**

```powershell
uv add fastapi              # Add REST API framework
uv add uvicorn              # ASGI server for FastAPI
uv add anthropic            # Anthropic API client
uv add "langchain-anthropic>=0.2.0"   # With version constraint
```

This updates `pyproject.toml` and `uv.lock`, and installs into `.venv`.

### Add a development-only dependency

```powershell
uv add --dev ruff           # Linter
uv add --dev black          # Formatter
uv add --dev mypy           # Type checker
```

Dev dependencies go under `[project.optional-dependencies]` or uv's dev group.

### Add an optional dependency group

For optional features (like `local-llm`), edit `pyproject.toml` or use:

```powershell
uv add --optional local-llm llama-cpp-python
```

Then install with:

```powershell
uv sync --extra local-llm
```

### Install all dependencies after pulling changes

If someone else added packages or you cloned the repo:

```powershell
uv sync --extra dev
```

Install with optional extras:

```powershell
uv sync --extra dev --extra local-llm
```

### Remove a dependency

```powershell
uv remove <package-name>
```

### Upgrade a dependency

```powershell
uv add <package-name> --upgrade
```

### List installed packages

```powershell
uv pip list
```

Or with activated venv:

```powershell
pip list
```

### Important: do not use plain `pip install` for project deps

| Avoid | Use instead |
|-------|-------------|
| `pip install pandas` | `uv add pandas` |
| `pip install -r requirements.txt` | `uv sync` |

Using `uv add` keeps `pyproject.toml` and `uv.lock` in sync for reproducible installs.

---

## Daily workflow cheat sheet

```powershell
# Open project
cd C:\MY_SPACE\AI_ML_GENAI

# Sync deps (after git pull or adding packages)
uv sync --extra dev

# --- Run without activating (recommended) ---
uv run streamlit run app.py          # Start UI
uv run python ingest.py              # Rebuild RAG index
uv run python mcp_server/server.py          # MCP server
uv run pytest                        # Tests

# --- Or activate first ---
.\.venv\Scripts\Activate.ps1
streamlit run app.py
deactivate

# --- Add a new package ---
uv add <package-name>
uv sync
```

---

## Related docs

- [Getting Started](getting-started.md) — first-time setup
- [User Manual](user-manual.md) — using each app mode
- [Configuration](configuration.md) — `.env` variables
- [Troubleshooting](troubleshooting.md) — common errors
