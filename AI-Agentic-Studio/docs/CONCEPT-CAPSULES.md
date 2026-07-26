# Concept Capsules

**Bite-sized learning cards** for every Generative AI and Agentic AI concept implemented in AI Agentic Studio.

- Read **one capsule at a time** during your journey.
- Each capsule: **what it is → why it matters → in this repo → try it**.
- For tables and checklists, see [CONCEPTS.md](CONCEPTS.md).
- For hands-on practice, see [LEARNING-GUIDE.md](LEARNING-GUIDE.md) labs.

---

## How to use this on your learning journey

| Phase | Capsules to read | Then do |
|-------|------------------|---------|
| **Week 1** | Track 1 (G1–G14) + R1–R5 | [LEARNING-PATH Week 1](LEARNING-PATH.md#week-1--generative-ai--rag-foundations) · Labs 1–2, 4 |
| **Week 2** | Track 2 (R6–R22) | [LEARNING-PATH Week 2](LEARNING-PATH.md#week-2--advanced-rag) · Labs 3, 5, 9 |
| **Week 3** | Track 3 (A1–A18) | [LEARNING-PATH Week 3](LEARNING-PATH.md#week-3--agentic-ai) · Labs 6–8, 12 |
| **Week 4** | Track 4 (P1–P12) | [LEARNING-PATH Week 4](LEARNING-PATH.md#week-4--production-patterns--code-confidence) · Labs 10–11 + capstone |

**Full day-by-day schedule:** [LEARNING-PATH.md](LEARNING-PATH.md)

---

# Track 1 — Generative AI fundamentals

---

### 💊 G1 · Generative AI

**One line:** AI that *creates* new text (or other content), not just classifies it.

**What:** Models like GPT, Claude, and Llama read your prompt and write a continuation — an answer, summary, code, etc.

**Why it matters:** This is the foundation of chatbots, copilots, and agents. Everything in this repo starts with a generative model.

**Analogy:** A very well-read assistant who completes your sentences — but only from patterns it learned, not from live facts unless you give it context (RAG).

**In this repo:** `agentic_studio/llm/` · default provider `echo` (offline) or `openai`, `anthropic`, etc.

**Try:** `studio ask "What is hybrid retrieval?"`

**See also:** G2 LLM, B1 RAG

---

### 💊 G2 · LLM (Large Language Model)

**One line:** A neural network trained on huge amounts of text to predict the next token.

**What:** The “brain” behind `studio ask`, `studio agent`, and chat. It does not browse the web or read your files unless you connect tools or RAG.

**Why it matters:** All answers and agent reasoning pass through an LLM. Choosing and routing providers is a core skill.

**Analogy:** A student who memorized millions of textbooks — strong at language, weak at knowing *your* private data without help.

**In this repo:** `llm/router.py`, `llm/providers/` · chain via `STUDIO_LLM_PROVIDERS`

**Try:** `studio doctor` → see active providers

**See also:** G3 Prompt, G8 Provider

---

### 💊 G3 · Prompt

**One line:** The full message you send the model — instructions + context + question.

**What:** A prompt can be a single user question or a stack of system + history + retrieved documents.

**Why it matters:** Better prompts → better answers. RAG and agents are mostly about *building* the right prompt automatically.

**Analogy:** The briefing you give a consultant before they speak.

**In this repo:** `rag/prompts.py` (RAG templates), agent system messages in `agents/react.py`

**Try:** Compare `studio search` (no LLM prompt for answer) vs `studio ask` (full RAG prompt)

**See also:** B1 RAG, A1 Agent

---

### 💊 G4 · Token

**One line:** A small piece of text the model reads or writes (often ~¾ of a word).

**What:** Models have limits measured in tokens. Long chats and big document context cost more tokens.

**Why it matters:** Explains billing, speed, and why memory summarization exists.

**Analogy:** Words chopped into syllables — the model’s alphabet.

**In this repo:** `Usage` in API/CLI JSON · `observability/metrics.py`

**Try:** `studio ask "..." --json` → check `usage.prompt_tokens`

**See also:** G5 Context window, A15 Summarizing memory

---

### 💊 G5 · Context window

**One line:** Maximum tokens the model can see in one request.

**What:** Includes system prompt, chat history, retrieved chunks, and the new question. Exceeding it truncates or fails.

**Why it matters:** You cannot paste a whole book into one call — you chunk, retrieve, and summarize.

**Analogy:** Desk space — only so many pages fit at once.

**In this repo:** `memory/summarizing.py` compacts old turns · chunking limits in `rag/chunking.py`

**Try:** Long chat in `studio ui` → memory eventually summarizes

**See also:** B3 Chunk, A14 Memory

---

### 💊 G6 · Temperature

**One line:** Knob for randomness in generation (0 = focused, higher = creative).

**What:** Low temperature for factual RAG; higher for brainstorming (not default here).

**Why it matters:** Same prompt can give different answers at different temperatures.

**In this repo:** `STUDIO_LLM_TEMPERATURE` in `.env` (default `0.2`)

**Try:** Set `0.0` vs `0.8`, run same `studio ask` twice (with non-echo provider for visible effect)

**See also:** G2 LLM

---

### 💊 G7 · Streaming

**One line:** Answer appears word-by-word instead of waiting for the full response.

**What:** Server-Sent Events (SSE) push tokens as they are generated.

**Why it matters:** Better UX for long answers; same underlying model.

**In this repo:** `pipeline.stream_answer()`, `/rag/stream`, UI Chat tab

**Try:** `studio ui` → Chat · or `curl -N POST .../rag/stream`

**See also:** P11 SSE

---

### 💊 G8 · Provider & failover

**One line:** Which API or local engine runs the LLM, with automatic backup if one fails.

**What:** Chain like `openai,anthropic,echo` — router tries each in order.

**Why it matters:** Production systems need reliability and vendor flexibility.

**In this repo:** `llm/router.py` · `STUDIO_LLM_PROVIDERS`

**Try:** `studio doctor` · set `STUDIO_LLM_PROVIDERS=echo` for offline

**See also:** G2 LLM

---

### 💊 G9 · Caching

**One line:** Store past LLM responses to avoid paying twice for the same question.

**What:** Exact match cache + optional semantic cache (similar questions).

**Why it matters:** Cuts cost and latency in repeated workloads.

**In this repo:** `llm/cache.py` · `STUDIO_CACHE_ENABLED`

**Try:** Run identical `studio ask` twice — second may be faster (with cache on)

**See also:** G2 LLM

---

### 💊 G10 · Structured output

**One line:** Force the model to return JSON matching a schema.

**What:** Used for plans, tool arguments, and evaluation — not free-form prose.

**Why it matters:** Agents need machine-readable decisions, not just chat.

**In this repo:** `llm/structured.py` · planner and tool-call parsing

**Try:** `studio agent --mode plan --json` → inspect structured steps

**See also:** A3 Tool / function calling

---

### 💊 G11 · Grounding

**One line:** Tie the answer to real sources, not model guesswork.

**What:** RAG retrieves documents first; the model is instructed to cite them.

**Why it matters:** Reduces hallucination on factual questions about *your* data.

**Analogy:** Open-book exam vs closed-book exam.

**In this repo:** Entire `rag/pipeline.py` · citations `[1]`, `[2]`

**Try:** `studio ask "What does BM25 catch?"` → note sources

**See also:** B1 RAG, G12 Hallucination

---

### 💊 G12 · Hallucination

**One line:** Model states something confident but wrong or unsupported.

**What:** Common when asking about private facts without retrieval.

**Why it matters:** Why RAG, citations, faithfulness metrics, and guardrails exist.

**In this repo:** Mitigated by RAG + `evaluation/metrics.py` faithfulness score

**Try:** `studio eval` → check faithfulness column · compare ask with/without ingest

**See also:** G11 Grounding, B22 Faithfulness

---

### 💊 G13 · Citation

**One line:** Marker linking an answer sentence to a source chunk.

**What:** e.g. `[1]` maps to the first retrieved passage in the sources list.

**Why it matters:** User can verify claims; required for trustworthy RAG.

**In this repo:** `RagAnswer` contexts · `SourceModel` in API

**Try:** `studio ask "..."` → match `[1]` to Sources list

**See also:** B1 RAG

---

### 💊 G14 · Multimodal (vision)

**One line:** Model input includes images as well as text.

**What:** Images encoded as data URLs or passed to vision-capable providers.

**Why it matters:** Diagrams, screenshots, and documents with layouts.

**In this repo:** `multimodal/vision.py` · `Message.images` in `core/types.py`

**Try:** Read `vision.py` · use with `openai` vision model when configured

**See also:** G2 LLM

---

# Track 2 — RAG (Retrieval-Augmented Generation)

---

### 💊 R1 · RAG

**One line:** Retrieve relevant documents, then generate an answer from them.

**What:** Three steps: **ingest** → **retrieve** → **generate**.

**Why it matters:** The standard pattern for “chat with your PDFs” and enterprise Q&A.

**Analogy:** Research paper: bibliography first, then write the essay.

**In this repo:** `rag/pipeline.py` · `studio ask`, `/rag/query`

**Try:** `studio ingest data/raw` then `studio ask "..."`

**See also:** R2 Ingest, R6 Dense retrieval

---

### 💊 R2 · Ingest

**One line:** Load files, chunk them, embed them, save to an index.

**What:** Turns raw documents into searchable pieces on disk (`var/index/`).

**Why it matters:** No ingest = empty corpus = no grounded answers.

**In this repo:** `rag/ingest.py`, `rag/loader.py`

**Try:** `studio ingest data/raw`

**See also:** R3 Chunk, R4 Embedding

---

### 💊 R3 · Chunk

**One line:** A small segment of a document used for search and citation.

**What:** Long PDFs are split (e.g. ~600 chars) with optional overlap.

**Why it matters:** Embeddings work on chunks, not whole books; chunk size affects recall and precision.

**Analogy:** Index cards instead of the whole textbook.

**In this repo:** `rag/chunking.py` · `STUDIO_CHUNK_SIZE`, `STUDIO_CHUNK_OVERLAP`

**Try:** After ingest, inspect `var/index/chunks.json`

**See also:** R5 Parent document

---

### 💊 R4 · Embedding

**One line:** Convert text into a vector of numbers capturing meaning.

**What:** Similar texts → vectors close together in space.

**Why it matters:** Powers semantic (dense) search — finds paraphrases, not just keywords.

**In this repo:** `rag/embeddings.py` · `hashing` (offline) or `sentence-transformers`

**Try:** `studio doctor` → `embedder` field

**See also:** R6 Dense retrieval, R7 Vector store

---

### 💊 R5 · Parent document retrieval

**One line:** Search on small chunks but show the model a larger surrounding passage.

**What:** Improves context without losing precise retrieval.

**Why it matters:** Small chunks match queries well but may lack surrounding sentences.

**In this repo:** `rag/chunking.py` · `STUDIO_PARENT_CHUNK_SIZE`

**Try:** Read retrieval handbook in `data/raw/`

**See also:** R3 Chunk

---

### 💊 R6 · Dense retrieval

**One line:** Find chunks whose embeddings are closest to the question embedding.

**What:** Also called semantic or vector search.

**Why it matters:** Catches meaning and paraphrase; misses rare exact codes sometimes.

**In this repo:** `rag/vector_store.py` · numpy or FAISS backend

**Try:** `studio search "semantic similarity"` → retriever includes `dense`

**See also:** R8 BM25, R9 Hybrid

---

### 💊 R7 · Vector store

**One line:** Database that stores chunk embeddings and supports similarity search.

**What:** Persisted under `var/index/` with metadata (source, page, etc.).

**Why it matters:** Central piece of any RAG system.

**In this repo:** `NumpyVectorStore`, `FaissVectorStore`

**Try:** `studio doctor` → corpus `chunks` count

**See also:** R4 Embedding

---

### 💊 R8 · BM25

**One line:** Classic keyword ranking — great for exact terms and rare words.

**What:** Lexical (sparse) retrieval complementing dense search.

**Why it matters:** Finds product IDs, error codes, and names embeddings may smooth over.

**Analogy:** Ctrl+F with smart scoring.

**In this repo:** `rag/lexical.py`

**Try:** `studio search "BM25 identifiers"` → `bm25` in retriever tag

**See also:** R9 Hybrid

---

### 💊 R9 · Hybrid retrieval

**One line:** Run dense + BM25, then merge results.

**What:** Default in this project (`STUDIO_HYBRID_ENABLED=true`).

**Why it matters:** Industry best practice for mixed query types.

**In this repo:** `rag/pipeline.py` retrieve loop

**Try:** Lab 3 in LEARNING-GUIDE (toggle hybrid off/on)

**See also:** R10 RRF

---

### 💊 R10 · RRF (Reciprocal Rank Fusion)

**One line:** Merge ranked lists using ranks only — no need to compare incompatible scores.

**What:** Score ≈ sum of 1/(k + rank) across retrievers.

**Why it matters:** Simple, strong fusion when BM25 and cosine scores live on different scales.

**In this repo:** `rag/fusion.py`

**Try:** `studio ask "What problem does reciprocal rank fusion solve?"`

**See also:** R9 Hybrid

---

### 💊 R11 · Reranking

**One line:** Re-order top candidates so the best chunks rise to the top.

**What:** Lexical, cross-encoder, or LLM-based rerankers.

**Why it matters:** Retrieval optimizes recall; reranking optimizes precision for the LLM context.

**Analogy:** Shortlist 30 résumés, then carefully rank the top 8.

**In this repo:** `rag/rerank.py` · `STUDIO_RERANKER`

**Try:** `STUDIO_RERANKER=lexical` (default) vs `cross-encoder` with `[retrieval]` installed

**See also:** R6 Dense retrieval

---

### 💊 R12 · Query transform

**One line:** Rewrite or expand the user question before search.

**What:** Strategies: `rewrite`, `multi-query`, `hyde`, `decompose`, or `none`.

**Why it matters:** One phrasing may miss relevant docs; variants improve recall.

**In this repo:** `rag/query_transform.py` · `STUDIO_QUERY_TRANSFORM` (default `multi-query`)

**Try:** `studio ask --json` → `queries_used` array

**See also:** R13 Multi-query, R14 HyDE

---

### 💊 R13 · Multi-query

**One line:** Generate several search queries from one user question.

**What:** Default query transform in this project.

**Why it matters:** Covers synonyms and aspect splits automatically.

**In this repo:** `QueryTransformer` with `multi-query`

**Try:** `studio search "RRF"` → note multiple queries in output

**See also:** R12 Query transform

---

### 💊 R14 · HyDE

**One line:** Hypothetical Document Embeddings — search using a fake answer embedding.

**What:** LLM writes a hypothetical passage; embed that for retrieval.

**Why it matters:** Helps when question and document wording differ greatly.

**In this repo:** `STUDIO_QUERY_TRANSFORM=hyde`

**Try:** Set in `.env`, re-run `studio search` on same question

**See also:** R12 Query transform

---

### 💊 R15 · Graph RAG

**One line:** Use an entity graph built from chunks to expand retrieval.

**What:** Entities co-occurring in text become linked nodes; query can hop neighbors.

**Why it matters:** Surfaces related concepts not in the top vector hits alone.

**In this repo:** `rag/graph_rag.py` · tool `graph_explore`

**Try:** `studio doctor` → `graph_entities`, `graph_edges` · agent tool `graph_explore`

**See also:** R6 Dense retrieval

---

### 💊 R16 · Metadata filter

**One line:** Restrict search to chunks matching properties (source, page, type).

**What:** e.g. only PDFs, or `source` contains `handbook`.

**Why it matters:** Multi-tenant and multi-collection corpora.

**In this repo:** Vector store `where` predicates · API `metadata_filter`

**Try:** API `POST /rag/query` with `"metadata_filter": {"source": {"$contains": "handbook"}}`

**See also:** R7 Vector store

---

### 💊 R17 · Conversational RAG

**One line:** Use chat history to interpret follow-up questions.

**What:** “How is it different?” needs prior turn context.

**Why it matters:** Real chat is multi-turn, not isolated questions.

**In this repo:** `rag/conversational.py` · `/chat`, UI Chat tab

**Try:** Lab 5 — ask two related questions in `studio ui`

**See also:** A14 Memory, G5 Context window

---

### 💊 R18 · Retrieval-only vs full RAG

**One line:** `search` shows chunks; `ask` also generates an answer.

**What:** Debugging retrieval without LLM noise.

**Why it matters:** When answers are wrong, check if retrieval or generation failed.

**In this repo:** `studio search` vs `studio ask` · `/rag/search` vs `/rag/query`

**Try:** Lab 2 in LEARNING-GUIDE

**See also:** R1 RAG

---

### 💊 R19 · Top-k and fetch-k

**One line:** How many chunks to return vs how many to consider before fusion/rerank.

**What:** `fetch_k` ≥ `top_k` — cast a wide net, then narrow.

**In this repo:** `STUDIO_RETRIEVAL_TOP_K`, `STUDIO_RETRIEVAL_FETCH_K`

**Try:** Lower `top_k` to 3 in `.env`, compare answer breadth

**See also:** R11 Reranking

---

### 💊 R20 · RAG evaluation metrics

**One line:** Numbers that score retrieval and answer quality on a golden dataset.

**What:** Faithfulness, relevance, precision, recall, citation quality, correctness.

**Why it matters:** Prove a config change actually helped.

**In this repo:** `evaluation/metrics.py`, `evaluation/runner.py`

**Try:** `studio eval` · read `reports/*.md`

**See also:** B22 Faithfulness (below), P8 Evaluation

---

### 💊 R21 · Faithfulness

**One line:** Are answer claims supported by retrieved text?

**What:** Core RAG quality metric — detects unsupported hallucination.

**In this repo:** `metrics.faithfulness()` · column in eval report

**Try:** `studio eval` → faithfulness column

**See also:** G12 Hallucination, R20

---

### 💊 R22 · Context precision & recall

**One line:** Precision = retrieved chunks on-topic; recall = reference facts were found.

**What:** Diagnoses “too much junk in context” vs “right answer not in index.”

**In this repo:** `context_precision`, `context_recall` in eval

**Try:** `studio eval --compare`

**See also:** R20

---

# Track 3 — Agentic AI

---

### 💊 A1 · Agent

**One line:** LLM + loop that uses tools until a task is done.

**What:** Not one shot — repeated think → act → observe.

**Why it matters:** Enables search, calculation, file ops, and multi-step research.

**Analogy:** Employee with a checklist and a phone to call departments (tools).

**In this repo:** `agents/react.py`, `planner.py`, `supervisor.py`

**Try:** `studio agent "Search the corpus for BM25 and summarise." --json`

**See also:** A2 ReAct, A3 Tool

---

### 💊 A2 · ReAct

**One line:** **Re**ason + **Act** — alternate thinking and tool use.

**What:** Default agent mode: `think` node → `act` node → loop.

**Why it matters:** Most common agent pattern in production copilots.

**In this repo:** `ToolCallingAgent` · `studio graph` for Mermaid diagram

**Try:** `studio agent "..." --mode react` · `studio graph`

**See also:** A1 Agent

---

### 💊 A3 · Tool (function calling)

**One line:** Named function the model can invoke with JSON arguments.

**What:** Calculator, search, RAG, HTTP, etc. — schema describes parameters.

**Why it matters:** Extends LLM from talk-only to *doing* things.

**In this repo:** `agents/tools/registry.py` · 15 registered tools

**Try:** `studio tools`

**See also:** A4 Tool schema, A12 RAG-as-tool

---

### 💊 A4 · Tool schema

**One line:** JSON description of tool name, parameters, types, and descriptions.

**What:** Auto-generated from Python type hints and docstrings here.

**Why it matters:** Model must know *how* to call each tool correctly.

**In this repo:** `infer_schema()` in `registry.py` · `GET /tools`

**Try:** `studio tools` · compare to function in `agents/tools/`

**See also:** A3 Tool

---

### 💊 A5 · Plan-execute agent

**One line:** Write a plan first, execute step by step, critique at the end.

**What:** Better for long tasks needing structure.

**Why it matters:** Reduces random tool thrashing on complex goals.

**In this repo:** `PlanExecuteAgent` · `--mode plan`

**Try:** `studio agent "Compare BM25 and dense retrieval" --mode plan --json`

**See also:** A6 Critic, A2 ReAct

---

### 💊 A6 · Critic

**One line:** Review step that checks draft output against the goal.

**What:** Part of plan-execute loop — can trigger revision.

**Why it matters:** Self-correction without human intervention.

**In this repo:** `agents/planner.py` critique node

**Try:** `--mode plan` with a multi-part question

**See also:** A5 Plan-execute

---

### 💊 A7 · Supervisor (multi-agent)

**One line:** Router agent delegates to specialist sub-agents.

**What:** e.g. research specialist vs compute specialist.

**Why it matters:** Smaller tool sets per agent → fewer mistakes.

**Analogy:** Manager assigning tasks to team members with different skills.

**In this repo:** `SupervisorAgent` · `--mode team`

**Try:** `studio agent "Research RRF and calculate 999*888" --mode team`

**See also:** A1 Agent

---

### 💊 A8 · StateGraph

**One line:** Engine that runs nodes and edges with shared state.

**What:** General orchestration — not only linear chat.

**Why it matters:** Checkpointing, branching, interrupts, and streaming per node.

**In this repo:** `agents/graph.py`

**Try:** `studio graph` · read `graph.py`

**See also:** A9 Checkpoint, A10 Interrupt

---

### 💊 A9 · Checkpoint

**One line:** Save agent state to disk so a run can resume.

**What:** SQLite or memory checkpointer stores messages, steps, usage.

**Why it matters:** Required for HITL pause/resume and long workflows.

**In this repo:** `agents/checkpoint.py`

**Try:** Approve a tool in UI — run resumes from checkpoint

**See also:** A11 HITL

---

### 💊 A10 · Interrupt & resume

**One line:** Pause the graph mid-run; continue after external input.

**What:** Status `interrupted` until approval or resume value provided.

**In this repo:** `graph.py` `Interrupt` · `POST /agent/approvals/{id}`

**Try:** Trigger `python_exec` or `write_file` (approval required)

**See also:** A11 HITL

---

### 💊 A11 · HITL (Human-in-the-loop)

**One line:** Human must approve dangerous tool calls before execution.

**What:** `requires_approval=True` on tools like `python_exec`, `write_file`.

**Why it matters:** Safety for code execution and file writes.

**In this repo:** `agents/hitl.py` · UI sidebar pending approvals

**Try:** Lab 8 · agent task that needs `write_file`

**See also:** P1 Guardrails, A10 Interrupt

---

### 💊 A12 · RAG-as-tool

**One line:** Agent searches your corpus via `rag_search` / `rag_answer` tools.

**What:** Connects agentic and generative tracks — agent decides *when* to retrieve.

**Why it matters:** Dynamic research during multi-step tasks.

**In this repo:** `agents/tools/rag_tools.py`

**Try:** Lab 6 in LEARNING-GUIDE

**See also:** R1 RAG, A3 Tool

---

### 💊 A13 · Parallel tool execution

**One line:** Run independent tool calls concurrently, preserve order in results.

**What:** Faster when model requests multiple tools at once.

**In this repo:** `registry.run_many()` · `STUDIO_AGENT_PARALLEL_TOOLS`

**Try:** `studio agent --json` → multiple tools in one `act` step

**See also:** A3 Tool

---

### 💊 A14 · Memory (conversation store)

**One line:** Persist chat messages per `thread_id` in SQLite.

**What:** Survives restarts; shared by API and UI.

**Why it matters:** Stateful assistants need durable history.

**In this repo:** `memory/store.py` · `var/memory.sqlite3`

**Try:** `/threads` API · Lab 5

**See also:** A15 Summarizing memory, R17 Conversational RAG

---

### 💊 A15 · Summarizing memory

**One line:** When history is too long, compress old turns into a summary.

**What:** Keeps recent messages verbatim + rolling summary of older ones.

**Why it matters:** Fits long conversations in the context window.

**In this repo:** `memory/summarizing.py`

**Try:** Long chat in UI → `GET /threads/{id}` may show `summary`

**See also:** G5 Context window

---

### 💊 A16 · Max steps

**One line:** Cap on agent loop iterations to prevent infinite runs.

**What:** Default 12 steps; status `max_steps_exceeded` if hit.

**In this repo:** `STUDIO_AGENT_MAX_STEPS` · API `max_steps`

**Try:** Very hard task with `max_steps: 2` via API

**See also:** A1 Agent

---

### 💊 A17 · Tool timeout & retry

**One line:** Kill hung tools; retry flaky ones automatically.

**What:** Protects agent from stuck network or bad code.

**In this repo:** `ToolRegistry.run()` · `STUDIO_AGENT_TOOL_TIMEOUT_S`

**Try:** Read `tests/test_tools.py` timeout test

**See also:** A3 Tool, P5 Sandbox

---

### 💊 A18 · Offline web search

**One line:** `web_search` uses local corpus when no internet API configured.

**What:** `STUDIO_SEARCH_PROVIDER=offline` — demos work without keys.

**Why it matters:** Same agent code in dev/test/prod with different backends.

**In this repo:** `agents/tools/web_search.py`

**Try:** `studio agent` task needing search — results show `local://` URLs

**See also:** A12 RAG-as-tool

---

# Track 4 — Safety, observability & production

---

### 💊 P1 · Guardrails

**One line:** Policy layer on input, output, tools, and retrieved context.

**What:** Central `GuardrailPolicy` at every trust boundary.

**Why it matters:** LLMs and agents are not safe by default for open user input.

**In this repo:** `guardrails/policy.py` · `STUDIO_GUARDRAILS_ENABLED`

**Try:** Lab 8 — unsafe prompt blocked

**See also:** P2 PII, P3 Moderation

---

### 💊 P2 · PII redaction

**One line:** Detect and mask emails, phones, cards, tokens in text.

**What:** Modes: `redact` (mask), `block` (reject), or `off`.

**Why it matters:** Compliance before sending data to hosted models or logs.

**In this repo:** `guardrails/pii.py` · `STUDIO_PII_MODE`

**Try:** Ask with fake email in question — check redaction in logs/output

**See also:** P1 Guardrails

---

### 💊 P3 · Moderation

**One line:** Block requests for harmful or dangerous content.

**What:** Local pattern rules; optional remote OpenAI moderation if keyed.

**In this repo:** `guardrails/moderation.py` · `STUDIO_MODERATION_MODE=block`

**Try:** Lab 8

**See also:** P1 Guardrails

---

### 💊 P4 · Prompt injection

**One line:** Malicious text in documents or input tries to override instructions.

**What:** Especially dangerous in RAG — untrusted PDFs become part of the prompt.

**Why it matters:** Agents that browse/fetch amplify the risk.

**In this repo:** `detect_injection()`, `sanitize_retrieved()` in `moderation.py`

**Try:** Read `clean_context` in `policy.py`

**See also:** R1 RAG, P1 Guardrails

---

### 💊 P5 · Sandbox

**One line:** Restricted environment for code and filesystem tools.

**What:** Python runs in subprocess with blocked imports; files confined to `var/sandbox/`.

**Why it matters:** Defense in depth — not a full security boundary, but limits accidents.

**In this repo:** `python_exec.py`, `filesystem.py`

**Try:** `write_file` only under sandbox via agent

**See also:** A11 HITL, P6 HTTP allowlist

---

### 💊 P6 · HTTP allowlist & SSRF protection

**One line:** Agent HTTP tool only calls approved hosts; blocks private IPs.

**What:** Prevents agent from probing internal networks or cloud metadata.

**In this repo:** `agents/tools/http.py` · `STUDIO_HTTP_ALLOWED_HOSTS`

**Try:** `http_request` to non-allowlisted host → error in tool result

**See also:** A3 Tool

---

### 💊 P7 · Observability (tracing)

**One line:** Record spans for each LLM call, tool run, and agent node.

**What:** JSONL file or optional LangSmith / OpenTelemetry export.

**Why it matters:** Debug “why did the agent do that?” in production.

**In this repo:** `observability/tracing.py` · `GET /traces`

**Try:** Run agent, then `curl http://localhost:8100/traces`

**See also:** P8 Metrics

---

### 💊 P8 · Metrics

**One line:** Counters and histograms — calls, latency, tokens, guardrail blocks.

**What:** In-process snapshot via `/metrics`.

**Why it matters:** SLOs, cost tracking, dashboards.

**In this repo:** `observability/metrics.py`

**Try:** `studio ui` → Observability tab · `GET /metrics`

**See also:** P7 Tracing

---

### 💊 P9 · Evaluation harness

**One line:** Run golden questions through pipeline and score results.

**What:** JSON + Markdown reports for humans and CI.

**In this repo:** `evaluation/runner.py` · `studio eval`

**Try:** Lab 9 · `reports/eval-*.md`

**See also:** R20, R21

---

### 💊 P10 · LLM-as-judge

**One line:** Second LLM scores answer quality when lexical metrics are not enough.

**What:** Optional with `studio eval --judge` and hosted provider.

**In this repo:** `evaluation/judge.py`

**Try:** `studio eval --judge` with `OPENAI_API_KEY` set

**See also:** P9 Evaluation

---

### 💊 P11 · SSE (streaming API)

**One line:** HTTP stream of events for token-by-token or step-by-step updates.

**What:** Used by `/rag/stream`, `/chat/stream`, `/agent/stream`.

**In this repo:** `api/streaming.py`

**Try:** Swagger or `curl -N` on stream endpoints

**See also:** G7 Streaming

---

### 💊 P12 · MCP (Model Context Protocol)

**One line:** Standard way to expose or consume tools across apps (e.g. Cursor).

**What:** This repo can **serve** its tools or **register** external MCP tools into agents.

**Why it matters:** Interop — one agent, many tool sources.

**In this repo:** `mcp_bridge/server.py`, `client.py`

**Try:** `studio mcp-serve` · read `mcp_bridge/config.json`

**See also:** A3 Tool

---

# Capsule index (quick lookup)

| ID | Name | Track |
|----|------|-------|
| G1–G14 | Generative AI | 1 |
| R1–R22 | RAG | 2 |
| A1–A18 | Agentic AI | 3 |
| P1–P12 | Safety & production | 4 |

**Total: 66 capsules** — one per major concept in AI Agentic Studio v1.

---

# Not in this repo (learn elsewhere)

| Topic | Where |
|-------|--------|
| Fine-tuning / training GPT | Parent `AI_ML_GENAI/MiniGPT/` |
| Image generation (DALL·E, SD) | External courses |
| Kubernetes / cloud deploy | [ROADMAP.md](ROADMAP.md) |
| Pinecone / Weaviate / pgvector | [ROADMAP.md](ROADMAP.md) |

---

**Next:** Follow **[LEARNING-PATH.md](LEARNING-PATH.md)** day by day, or pick **G1** and read one capsule per session.
