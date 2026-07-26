# MiniGPT (merged project)

The **MiniGPT** project is integrated into this workspace under `minigpt/`. It complements the main stack (RAG, agents, MCP, API, Streamlit) with **from-scratch GPT** implementations and **Llama 3 fine-tuning**.

## Location

```
AI_ML_GENAI/
├── rag/              # RAG pipeline
├── agents/           # AI agents
├── chat/             # Chatbot & assistant
├── api/              # FastAPI + Swagger
├── mcp_server/       # MCP tools
├── app.py            # Streamlit UI
└── minigpt/          # ← MiniGPT (merged from Downloads)
    ├── common/       # Shared tokenizer & data
    ├── pytorch/      # PyTorch MiniGPT
    ├── tensorflow/   # TensorFlow MiniGPT
    └── finetune/     # Llama 3 LoRA / QLoRA fine-tuning
```

## What MiniGPT includes

| Module | Purpose |
|--------|---------|
| `minigpt/common/` | Shared tokenizer, `song.txt` data, learning docs |
| `minigpt/pytorch/` | Train & chat with tiny GPT (PyTorch) |
| `minigpt/tensorflow/` | Same model in TensorFlow/Keras |
| `minigpt/finetune/` | Fine-tune Llama 3 on custom data (e.g. medical SFT) |

## Quick start (PyTorch MiniGPT)

From repo root:

```powershell
cd C:\MY_SPACE\AI_ML_GENAI

# Train tiny GPT on Twinkle Twinkle Little Star
uv run python minigpt/pytorch/train.py

# Chat in terminal
uv run python minigpt/pytorch/chat.py
```

Expected chat:

```
You: twinkle
MiniGPT: twinkle little star how i wonder what you are ...
```

## TensorFlow MiniGPT

```powershell
uv sync --extra minigpt-tensorflow
uv run python minigpt/tensorflow/train.py
uv run python minigpt/tensorflow/chat.py
```

## Llama 3 fine-tuning (finetune/)

Requires **NVIDIA GPU + CUDA** for QLoRA training.

```powershell
uv sync --extra minigpt-finetune
cd minigpt/finetune
copy .env.example .env
uv run python setup_check.py
uv run python train.py
uv run python inference.py
```

Full manual: [`minigpt/finetune/COMPLETE_MANUAL.md`](minigpt/finetune/COMPLETE_MANUAL.md)

## Optional dependency groups

| Extra | Command | For |
|-------|---------|-----|
| (core) | `uv sync --extra dev` | PyTorch MiniGPT (torch already included) |
| `minigpt-tensorflow` | `uv sync --extra minigpt-tensorflow` | TensorFlow MiniGPT |
| `minigpt-finetune` | `uv sync --extra minigpt-finetune` | Llama 3 LoRA / QLoRA |

## How MiniGPT fits the unified project

| Capability | Main project | MiniGPT |
|------------|--------------|---------|
| Chat with documents (RAG) | `app.py`, `/rag/query` API | — |
| Cloud / GGUF LLM | `rag/llm.py`, `.env` | — |
| Agents & MCP | `agents/`, `mcp_server/` | — |
| **Learn GPT architecture** | — | `minigpt/pytorch/`, docs |
| **Train tiny model from scratch** | — | `minigpt/pytorch/train.py` |
| **Fine-tune Llama 3 (LoRA)** | — | `minigpt/finetune/` |
| REST API + Swagger | `api_server.py` | — |
| Streamlit UI | `app.py` | Terminal `chat.py` |

Use **MiniGPT** to learn and experiment with model internals; use the **main stack** for production-style RAG, agents, and APIs.

## Learning docs (inside minigpt/)

| Topic | Path |
|-------|------|
| 10 steps to build an LLM | `minigpt/common/HOW_TO_BUILD_AN_LLM.md` |
| PyTorch MiniGPT | `minigpt/pytorch/README.md` |
| TensorFlow MiniGPT | `minigpt/tensorflow/README.md` |
| PyTorch vs TensorFlow | `minigpt/common/PYTORCH_VS_TENSORFLOW.md` |
| MiniGPT vs ChatGPT | `minigpt/common/COMPARISON.md` |
| Fine-tuning manual | `minigpt/finetune/COMPLETE_MANUAL.md` |

## Original project

Merged from: `C:\Users\nila0425\Downloads\MINIGPT`

## Related docs

- [Models Guide](models.md)
- [REST API](api.md)
- [Run & Dependencies](run-and-dependencies.md)
- [MiniGPT README](minigpt/README.md)
