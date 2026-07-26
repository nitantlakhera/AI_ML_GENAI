# REST API (Swagger)

FastAPI REST API with interactive **Swagger UI** for all major features.

## Start the API server

```powershell
cd C:\MY_SPACE\AI_ML_GENAI

# Option 1 — helper script (reload on port 8080)
uv run python api_server.py

# Option 2 — uvicorn directly
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8080
```

## Swagger UI

Open in your browser:

| URL | Description |
|-----|-------------|
| **http://localhost:8080/docs** | Swagger UI (interactive) |
| **http://localhost:8080/redoc** | ReDoc (alternative docs) |
| **http://localhost:8080/openapi.json** | OpenAPI schema (JSON) |

Use Swagger to explore endpoints, send test requests, and view request/response schemas.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | API and vector DB status |
| `POST` | `/chat` | Chatbot with optional `session_id` |
| `POST` | `/assistant` | AI assistant (optional RAG) |
| `POST` | `/rag/query` | RAG Q&A with sources |
| `POST` | `/agent` | Agent with tools |
| `POST` | `/ingest` | Rebuild FAISS index from `data/raw/` |
| `DELETE` | `/sessions/{session_id}` | Clear chat/assistant session |

## Example requests

### Health

```powershell
curl http://localhost:8080/health
```

### RAG query

```powershell
curl -X POST http://localhost:8080/rag/query `
  -H "Content-Type: application/json" `
  -d "{\"question\": \"What does the sample document say?\"}"
```

### Chat (with session)

```powershell
curl -X POST http://localhost:8080/chat `
  -H "Content-Type: application/json" `
  -d "{\"message\": \"Hello!\", \"session_id\": \"my-session-1\"}"
```

### Assistant

```powershell
curl -X POST http://localhost:8080/assistant `
  -H "Content-Type: application/json" `
  -d "{\"question\": \"Help me understand RAG\", \"use_rag\": true}"
```

### Agent

```powershell
curl -X POST http://localhost:8080/agent `
  -H "Content-Type: application/json" `
  -d "{\"task\": \"Calculate 20% of 150\"}"
```

### Ingest documents

```powershell
curl -X POST http://localhost:8080/ingest
```

## Prerequisites

Same as the main app:

1. `uv sync --extra dev`
2. Configure LLM in `.env` (OpenAI API or local GGUF) — see [Models Guide](models.md)
3. Run `uv run python ingest.py` before RAG endpoints

## Project layout

```
api/
├── main.py      # FastAPI app + routes
└── schemas.py   # Pydantic request/response models
api_server.py    # Entry point to start server
```

## Related docs

- [Health Check & Setup](health-check-and-setup.md)
- [Run & Dependencies](run-and-dependencies.md)
- [User Manual](user-manual.md)
