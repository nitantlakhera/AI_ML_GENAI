# Flow Diagrams

Step-by-step flows for each major capability. Each section includes a PNG diagram and a Mermaid flowchart (for editors that support it).

---

## 1. Overall application flow

```mermaid
flowchart TD
    A[User] --> B[app.py Streamlit UI]
    B --> C{Select Mode}
    C --> D[Chatbot]
    C --> E[AI Assistant]
    C --> F[RAG Q&A]
    C --> G[Agent]
    D --> H[chat/bot.py]
    E --> I[chat/assistant.py]
    F --> J[rag/chain.py]
    G --> K[agents/executor.py]
    H --> L[rag/llm.py]
    I --> L
    I --> J
    J --> M[vector_db FAISS]
    K --> L
    K --> N[agents/tools.py]
```

---

## 2. RAG pipeline flow

![RAG pipeline flow](images/rag-flow.png)

```mermaid
flowchart LR
    A[data/raw documents] --> B[ingest.py]
    B --> C[loader.py]
    C --> D[splitter.py]
    D --> E[embeddings.py]
    E --> F[vector_store.py]
    F --> G[vector_db/ FAISS]

    H[User question] --> I[retriever.py]
    G --> I
    I --> J[chain.py + prompt.py]
    J --> K[llm.py]
    K --> L[Answer + Sources]
```

### RAG ingestion steps

| Step | File | Action |
|------|------|--------|
| 1 | `ingest.py` | Entry point |
| 2 | `loader.py` | Load PDF/TXT/MD from `data/raw/` |
| 3 | `splitter.py` | Split into chunks (size/overlap from `.env`) |
| 4 | `embeddings.py` | Encode chunks with sentence-transformers |
| 5 | `vector_store.py` | Build and save FAISS index |

### RAG query steps

| Step | File | Action |
|------|------|--------|
| 1 | User | Submits question in RAG Q&A mode |
| 2 | `retriever.py` | Embed query, search top-K chunks |
| 3 | `prompt.py` | Inject context into template |
| 4 | `chain.py` | Run RetrievalQA chain |
| 5 | `llm.py` | Generate answer |
| 6 | UI | Display answer and source files |

---

## 3. Agent flow

![Agent flow](images/agent-flow.png)

```mermaid
flowchart TD
    A[User task] --> B[agents/executor.py]
    B --> C[agents/base.py build agent]
    C --> D[LLM receives task]
    D --> E{Need tool?}
    E -->|Yes| F[agents/tools.py]
    F --> G[Tool result]
    G --> D
    E -->|No| H[Final response]
    D --> I{Max iterations?}
    I -->|Exceeded| H
```

### Agent loop

1. User submits task via Agent mode
2. `executor.py` builds agent from `base.py`
3. LLM decides whether to answer directly or call a tool
4. If tool needed: `tools.py` runs calculator or word_count
5. Tool result returns to LLM
6. Loop until final answer or `AGENT_MAX_ITERATIONS`

---

## 4. MCP flow

![MCP flow](images/mcp-flow.png)

```mermaid
flowchart LR
    subgraph Client
        A[Cursor / MCP Client]
    end
    subgraph Server
        B[mcp_server/server.py FastMCP]
        C[mcp_server/tools/]
    end
    A <-->|stdio| B
    B --> C
    D[mcp_server/config.json] --> A
```

### MCP server lifecycle

1. Client reads `mcp_server/config.json`
2. Client spawns `uv run python mcp_server/server.py`
3. Server registers tools (`greet`, `word_count`)
4. Client calls `list_tools()` or `call_tool(name, args)`
5. Server executes and returns result

### MCP client usage

`mcp_server/client.py` can connect to **external** MCP servers:

```python
from mcp.client import list_tools
tools = await list_tools(["uv", "run", "python", "mcp_server/server.py"])
```

---

## 5. Chatbot & assistant flow

```mermaid
flowchart TD
    A[User message] --> B{Mode}
    B -->|Chatbot| C[chat/bot.py]
    B -->|Assistant| D[chat/assistant.py]
    C --> E[memory.py load history]
    E --> F[Build prompt + history]
    F --> G[llm.py]
    G --> H[memory.py save context]
    H --> I[Response]
    D --> J{RAG enabled?}
    J -->|Yes| K[rag/chain.py]
    J -->|No| F
    K --> I
```

---

## 6. Document ingestion decision flow

```mermaid
flowchart TD
    A[New documents added?] -->|Yes| B[Run ingest.py]
    A -->|No| C[Use existing vector_db]
    B --> D{Files in data/raw?}
    D -->|No| E[Error: no documents]
    D -->|Yes| F[Build FAISS index]
    F --> G[Ready for RAG modes]
```

---

## 7. LLM selection flow

```mermaid
flowchart TD
    A[rag/llm.py get_llm] --> B{USE_API_LLM?}
    B -->|true| C[ChatOpenAI API]
    B -->|false| D[llama-cpp-python LlamaCpp]
    D --> E{Installed?}
    E -->|No| F[Error: install local-llm extra]
    E -->|Yes| G[Load GGUF from models/]
```
