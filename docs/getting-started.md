# Getting Started

This guide walks you through setting up the AI / ML / GenAI workspace from scratch.

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Python | 3.10 or higher |
| uv | Package manager ([install guide](https://docs.astral.sh/uv/)) |
| Disk space | ~2 GB for dependencies; more for local models |
| API key (optional) | OpenAI key if using `USE_API_LLM=true` |

## Step 1: Install uv

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal or add uv to PATH:

```powershell
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
```

## Step 2: Clone / open project

```powershell
cd C:\MY_SPACE\AI_ML_GENAI
```

## Step 3: Install dependencies

```powershell
uv sync --extra dev
```

This creates `.venv/` and installs all core packages.

**Optional — local GGUF models:**

```powershell
uv sync --extra local-llm
```

Requires C++ build tools on Windows. Skip this if you use an API LLM.

## Step 4: Configure environment

```powershell
copy .env.example .env
```

Edit `.env` for your setup. See [Configuration Reference](configuration.md).

**Minimum for API usage:**

```env
USE_API_LLM=true
OPENAI_API_KEY=sk-your-key-here
```

## Step 5: Add documents (for RAG)

Place files in `data/raw/`:

- PDF (`.pdf`)
- Text (`.txt`)
- Markdown (`.md`)

## Step 6: Build vector index

```powershell
uv run python ingest.py
```

Output example: `Indexed 142 chunks into vector_db`

## Step 7: Run the application

```powershell
uv run streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

## Step 8: Verify installation

```powershell
uv run pytest
```

All tests should pass.

## Run, activate, and add dependencies

For daily use — activating `.venv`, running all commands, and adding packages with `uv` — see **[Run, Activate & Dependencies](run-and-dependencies.md)**.

Quick reference:

```powershell
# Activate (optional — uv run works without activation)
.\.venv\Scripts\Activate.ps1

# Run app
uv run streamlit run app.py

# Add a new package
uv add <package-name>
```

## What to do next

| Goal | Read |
|------|------|
| Run & activate venv | [Run, Activate & Dependencies](run-and-dependencies.md) |
| Models (get, configure, use) | [Models Guide](models.md) |
| Use the UI | [User Manual](user-manual.md) |
| Understand design | [Architecture](architecture.md) |
| See process flows | [Flow Diagrams](flow-diagrams.md) |
| Fix issues | [Troubleshooting](troubleshooting.md) |
| Verify everything works | [Health Check & Setup Status](health-check-and-setup.md) |
