# Roadmap

Optional enhancements **beyond** the current v1 implementation. Nothing listed here is built yet; see [IMPLEMENTATION-STATUS.md](IMPLEMENTATION-STATUS.md) for what exists today.

---

## Near-term (v1.1)

| Item | Rationale |
|------|-----------|
| Docker Compose profile | One-command API + UI + optional Ollama |
| pgvector / Qdrant vector backend | Swap local numpy/FAISS for a server vector DB |
| Async agent streaming in plan/team modes | Today only ReAct streams over SSE |
| Richer golden eval sets per domain | Starter set ships with 5 cases |

## Medium-term (v1.x)

| Item | Rationale |
|------|-----------|
| RBAC and multi-tenant threads | API keys are flat; no per-user isolation |
| Agent run replay from checkpoints | Inspect and resume any historical step |
| Built-in A/B experiment runner | Compare retrieval configs in production traffic |
| Document upload UI with preview | Streamlit ingest tab is basic |

## Long-term (v2+)

| Item | Rationale |
|------|-----------|
| Distributed agent workers | Scale tool execution beyond one process |
| Fine-tuning integration hook | Bridge to parent repo MiniGPT workflows |
| Voice / real-time multimodal | Beyond static image helpers |
| Hosted moderation model adapter | Pluggable safety classifiers |

---

## Contributing

When adding a feature:

1. Implement in the appropriate `agentic_studio/` module.
2. Add configuration to `settings.py` and `.env.example` if needed.
3. Expose via CLI and/or API if user-facing.
4. Update [IMPLEMENTATION-STATUS.md](IMPLEMENTATION-STATUS.md) — not this file — when the feature ships.
