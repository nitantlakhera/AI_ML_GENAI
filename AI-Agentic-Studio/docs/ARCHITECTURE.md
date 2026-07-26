# Architecture

System design for **AI-Agentic-Studio** v1 — reflecting only what is implemented today.

> **New to AI?** See **[LEARNING-GUIDE.md](LEARNING-GUIDE.md)** for a simpler four-layer diagram and step-by-step call flows (`studio ask`, `studio agent`, `studio ingest`, chat).

---

## High-level system diagram

![High-level system architecture](diagrams/architecture-high-level-system.png)

```mermaid
flowchart TB
    subgraph Interfaces["Interfaces"]
        CLI["CLI · studio"]
        API["FastAPI REST + SSE"]
        UI["Streamlit UI"]
        MCP["MCP server / client"]
    end

    subgraph Orchestration["Orchestration layer"]
        AG["Agents<br/>ReAct · Plan · Supervisor"]
        SG["StateGraph engine"]
        HITL["HITL approvals"]
        TR["Tool registry"]
    end

    subgraph Generative["Generative layer"]
        RAG["RAG pipeline"]
        LLM["LLM router"]
        MEM["Memory store"]
    end

    subgraph Safety["Safety & ops"]
        GR["Guardrails"]
        OBS["Observability"]
        EV["Evaluation"]
    end

    subgraph Storage["Local persistence"]
        IDX["Vector index<br/>numpy / FAISS"]
        SQL["SQLite<br/>memory · cache · jobs · checkpoints"]
    end

    CLI --> AG
    CLI --> RAG
    API --> AG
    API --> RAG
    API --> MEM
    UI --> AG
    UI --> RAG
    MCP --> TR

    AG --> SG
    SG --> HITL
    SG --> TR
    TR --> GR
    AG --> LLM
    RAG --> LLM
    RAG --> IDX
    RAG --> GR
    MEM --> SQL
    LLM --> SQL
    SG --> SQL
    HITL --> SQL
    AG --> OBS
    RAG --> OBS
    LLM --> OBS
    EV --> RAG
```

---

## Layer responsibilities

| Layer | Package | Responsibility |
|-------|---------|----------------|
| **Core** | `agentic_studio/core/` | Shared types, errors, settings |
| **LLM** | `agentic_studio/llm/` | Provider abstraction, routing, caching, structured output |
| **RAG** | `agentic_studio/rag/` | Ingestion, retrieval stack, grounded generation |
| **Agents** | `agentic_studio/agents/` | Graph engine, agent modes, built-in tools |
| **Memory** | `agentic_studio/memory/` | Durable threads and summarizing context window |
| **Guardrails** | `agentic_studio/guardrails/` | PII, moderation, injection, tool policy |
| **Observability** | `agentic_studio/observability/` | Logs, metrics, traces |
| **Evaluation** | `agentic_studio/evaluation/` | Metrics, datasets, runner |
| **API** | `agentic_studio/api/` | HTTP surface, auth, jobs, streaming |
| **UI** | `agentic_studio/ui/` | Streamlit playground |
| **MCP** | `agentic_studio/mcp_bridge/` | External tool interoperability |

---

## RAG pipeline architecture

![RAG pipeline architecture](diagrams/architecture-rag-pipeline.png)

```mermaid
flowchart TD
    IN[Ingest files / texts] --> LOAD[Document loaders]
    LOAD --> CHUNK[Chunking<br/>recursive · semantic · markdown]
    CHUNK --> EMB[Embed chunks]
    EMB --> VS[(Vector store)]
    CHUNK --> BM[(BM25 index)]
    CHUNK --> KG[(Knowledge graph)]

    Q[User question] --> QT[Query transform<br/>rewrite · multi-query · HyDE · decompose]
    QT --> DS[Dense search]
    QT --> LS[BM25 search]
    DS --> FU[RRF / weighted fusion]
    LS --> FU
    FU --> RR[Rerank<br/>lexical · cross-encoder · LLM]
    RR --> GR_R[Graph RAG expand]
    GR_R --> CTX[Top-k contexts]
    CTX --> SAN[Guardrail: sanitize context]
    SAN --> GEN[LLM generation]
    GEN --> OUT[Answer + citations]
    GEN --> GRO[Guardrail: output check]
```

### Retrieval configuration (environment)

| Variable | Default | Effect |
|----------|---------|--------|
| `STUDIO_HYBRID_ENABLED` | `true` | Combine dense + BM25 |
| `STUDIO_GRAPH_RAG_ENABLED` | `true` | Entity graph expansion |
| `STUDIO_RERANKER` | `lexical` | Reranking strategy |
| `STUDIO_QUERY_TRANSFORM` | `multi-query` | Query expansion |
| `STUDIO_RETRIEVAL_TOP_K` | `8` | Final context count |

---

## LLM router architecture

![LLM router architecture](diagrams/architecture-llm-router.png)

```mermaid
flowchart LR
    REQ[Request] --> CACHE{Cache hit?}
    CACHE -->|yes| RES[Return cached]
    CACHE -->|no| P1[Provider 1]
    P1 -->|fail| P2[Provider 2]
    P2 -->|fail| PN[Provider N]
    P1 -->|ok| MET[Record metrics + trace]
    P2 -->|ok| MET
    PN -->|ok| MET
    PN -->|all fail| ERR[AllProvidersFailed]
    MET --> STORE[Write cache]
    STORE --> RES2[Response]
```

Provider chain is set by `STUDIO_LLM_PROVIDERS` (comma-separated). Default: `echo` (offline).

Implemented providers:

| Provider | Backend | Offline? |
|----------|---------|----------|
| `echo` | Deterministic extractive answers | Yes |
| `scripted` | Fixed response script (tests) | Yes |
| `openai` | OpenAI-compatible HTTP API | No |
| `anthropic` | Anthropic Messages API | No |
| `gemini` | Google Gemini API | No |
| `ollama` | Local Ollama HTTP | Local |
| `llamacpp` | GGUF via llama-cpp-python | Local |

---

## Agent architecture

### StateGraph engine

![StateGraph engine](diagrams/architecture-stategraph.png)

```mermaid
stateDiagram-v2
    [*] --> Node1
    Node1 --> Node2: edge
    Node1 --> Node3: conditional
    Node2 --> Node3
    Node3 --> Interrupt: approval required
    Interrupt --> Node3: approved
    Interrupt --> [*]: rejected
    Node3 --> [*]: END
```

Features implemented:

- Named nodes with state reducers (e.g. `add_messages`)
- Conditional edges based on state
- SQLite or in-memory checkpointing
- Interrupt / resume for human-in-the-loop
- `stream()` yields per-node events

### Agent modes

![Agent modes](diagrams/architecture-agent-modes.png)

```mermaid
flowchart TB
    TASK[Task] --> MODE{mode}

    MODE -->|react| R1[think node]
    R1 --> R2[act node · parallel tools]
    R2 --> R1
    R2 --> R3[END]

    MODE -->|plan| P1[plan]
    P1 --> P2[execute step]
    P2 --> P3[critique]
    P3 --> P2
    P3 --> P4[END]

    MODE -->|team| S1[supervisor routes]
    S1 --> S2[research specialist]
    S1 --> S3[compute specialist]
    S2 --> S1
    S3 --> S1
    S1 --> S4[END]
```

### Tool execution flow

![Tool execution flow](diagrams/architecture-tool-execution.png)

```mermaid
sequenceDiagram
    participant Agent
    participant Guardrails
    participant Registry
    participant HITL
    participant Tool

    Agent->>Guardrails: check_tool(name, args)
    alt blocked
        Guardrails-->>Agent: refusal
    else allowed + requires approval
        Agent->>HITL: pause run
        HITL-->>Agent: approved / rejected
    end
    Agent->>Registry: run(call)
    Registry->>Tool: invoke with timeout
    Tool-->>Registry: result / error
    Registry-->>Agent: ToolResult
```

---

## Guardrail boundaries

![Guardrail boundaries](diagrams/architecture-guardrail-boundaries.png)

```mermaid
flowchart LR
    USER[User input] --> GI[check_input]
    GI --> RAG_IN[RAG / agent]
    RET[Retrieved text] --> GC[clean_context]
    GC --> RAG_IN
    RAG_IN --> GO[check_output]
    GO --> USER_OUT[User response]
    RAG_IN --> GT[check_tool]
    GT --> TOOL[Tool execution]
```

| Boundary | Checks |
|----------|--------|
| Input | Length, moderation, injection hints, PII redact/block |
| Context | Instruction defanging in retrieved documents |
| Output | Moderation, PII redaction |
| Tool | Allowlist, blocked tools, argument moderation |

---

## API architecture

![API architecture](diagrams/architecture-api.png)

```mermaid
flowchart TB
    CLIENT[HTTP client] --> MW[Rate limit middleware]
    MW --> AUTH{API key set?}
    AUTH -->|yes| KEY[require_api_key]
    AUTH -->|no| HANDLER[Route handler]
    KEY --> HANDLER
    HANDLER --> BG[BackgroundTasks / SSE]
    HANDLER --> CORE[Studio modules]
    BG --> JOBS[(jobs.sqlite3)]
```

Key route groups:

- **System**: `/health`, `/providers`, `/metrics`, `/traces`, `/jobs`
- **RAG**: `/ingest`, `/rag/query`, `/rag/search`, `/rag/stream`
- **Chat**: `/chat`, `/chat/stream`, `/threads`
- **Agents**: `/agent`, `/agent/stream`, `/agent/approvals`
- **Evaluation**: `/eval/run`

---

## Data flow: end-to-end RAG question

![End-to-end RAG query sequence](diagrams/architecture-rag-query-sequence.png)

```mermaid
sequenceDiagram
    participant User
    participant API
    participant Policy
    participant Pipeline
    participant Router
    participant Index

    User->>API: POST /rag/query
    API->>Policy: check_input(question)
    API->>Pipeline: answer(question)
    Pipeline->>Pipeline: transform query
    Pipeline->>Index: dense + BM25 search
    Index-->>Pipeline: ranked chunks
    Pipeline->>Pipeline: fuse + rerank + graph
    Pipeline->>Policy: clean_context(chunks)
    Pipeline->>Router: generate(prompt)
    Router-->>Pipeline: answer text
    Pipeline->>Policy: check_output(answer)
    Pipeline-->>API: RagAnswer + sources
    API-->>User: JSON response
```

---

## Directory → responsibility map

```
agentic_studio/
├── core/           Types, errors, settings
├── llm/            Providers, router, cache
├── rag/            Full retrieval + generation stack
├── agents/         Graph, agents, tools
├── memory/         Conversation persistence
├── guardrails/     Safety policy
├── observability/  Logs, metrics, traces
├── evaluation/     Quality measurement
├── api/            HTTP interface
├── ui/             Streamlit playground
├── mcp_bridge/     MCP interoperability
└── cli.py          Command-line entry
```

---

## Design principles (as implemented)

1. **Offline-first** — `echo` + `hashing` work with zero network and zero keys.
2. **Single configuration surface** — all behaviour driven by `settings.py` / `.env`.
3. **Layered modularity** — RAG, agents, and LLM are independent; agents call RAG via tools.
4. **Defence in depth** — guardrails at input, context, output, and tool boundaries.
5. **Observable by default** — every LLM call, tool run, and agent step is traced and metered.
6. **Testable** — pytest fixtures isolate state; no shared disk between tests.
