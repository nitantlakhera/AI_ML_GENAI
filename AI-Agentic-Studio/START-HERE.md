# START HERE

One-page guide: **install → run → use** AI Agentic Studio. No API keys required.

**Project folder:** `C:\MY_SPACE\AI_ML_GENAI\AI-Agentic-Studio`

---

## 1. Install (first time only)

Open **PowerShell** and run:

```powershell
cd C:\MY_SPACE\AI_ML_GENAI\AI-Agentic-Studio

python -m venv .venv
.venv\Scripts\activate

pip install -e ".[dev,ui]"
```

Optional: copy settings file (defaults work without it):

```powershell
copy .env.example .env
```

Check install:

```powershell
studio doctor
```

You should see `"provider": "echo"` and a list of tools.

---

## 2. Index documents (do once, or after adding files)

```powershell
studio ingest data/raw
```

---

## 3. Run — daily commands

Activate the environment if you opened a new terminal:

```powershell
cd C:\MY_SPACE\AI_ML_GENAI\AI-Agentic-Studio
.venv\Scripts\activate
```

| What you want | Command |
|---------------|---------|
| Ask a question (with sources) | `studio ask "What does BM25 catch?"` |
| Search only (no answer) | `studio search "reciprocal rank fusion"` |
| Run an agent | `studio agent "Search the corpus for RRF and summarise."` |
| List tools | `studio tools` |
| Run evaluation | `studio eval` |
| Check setup | `studio doctor` |

**Agent modes:**

```powershell
studio agent "your task" --mode react    # default · tool loop
studio agent "your task" --mode plan     # plan → execute → critique
studio agent "your task" --mode team     # supervisor + specialists
```

**JSON output (see agent steps):**

```powershell
studio agent "your task" --json
```

---

## 4. Run — web interfaces

### Streamlit UI (visual playground)

```powershell
studio ui
```

Open **http://localhost:8501** — tabs: Chat, Retrieval, Agents, Ingest, Eval, Observability.

### REST API (Swagger)

```powershell
studio serve
```

Open **http://localhost:8100/docs** — try `POST /rag/query` with:

```json
{ "question": "What is hybrid retrieval?" }
```

---

## 5. Typical first session

```powershell
.venv\Scripts\activate
studio ingest data/raw
studio ask "What does BM25 catch that dense retrieval misses?"
studio search "BM25"
studio agent "Use rag_search to find BM25 and explain it briefly." --json
studio ui
```

---

## 6. Troubleshooting

| Problem | Fix |
|---------|-----|
| `studio` not found | Run `.venv\Scripts\activate` and `pip install -e ".[dev]"` |
| No answer / no sources | Run `studio ingest data/raw` first |
| Streamlit error | `pip install -e ".[ui]"` |
| Start fresh index | Delete `var\` folder, then `studio ingest data/raw` again |

---

## 7. Learn more (full docs)

| Read this | When |
|-----------|------|
| **[docs/LEARNING-PATH.md](docs/LEARNING-PATH.md)** | **4-week day-by-day plan** — start here for structured learning |
| **[docs/CONCEPT-CAPSULES.md](docs/CONCEPT-CAPSULES.md)** | One concept at a time — capsules with analogies |
| **[docs/CONCEPTS.md](docs/CONCEPTS.md)** | All terms in tables (quick lookup) |
| **[docs/LEARNING-GUIDE.md](docs/LEARNING-GUIDE.md)** | Beginner path, diagrams, **12 labs** |
| **[docs/USER-GUIDE.md](docs/USER-GUIDE.md)** | Detailed how-to for every feature |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | System design and call flows (PNG + Mermaid) |
| **[docs/diagrams/](docs/diagrams/)** | Standalone PNG images — open folder to view diagrams offline |
| **[README.md](README.md)** | Project overview and coverage summary |

---

## 8. Optional: use OpenAI instead of offline mode

Edit `.env`:

```env
STUDIO_LLM_PROVIDERS=openai,echo
OPENAI_API_KEY=sk-your-key-here
```

Restart the terminal, then run commands as usual. `echo` remains as fallback.

---

**Next step:** open **[docs/LEARNING-PATH.md](docs/LEARNING-PATH.md)** and begin **Day 0**, or jump to **Lab 1** in [docs/LEARNING-GUIDE.md](docs/LEARNING-GUIDE.md#lab-1--your-first-grounded-answer).
