from pathlib import Path

from chat.memory import get_memory
from config.settings import VECTOR_DB_DIR
from rag.chain import build_rag_chain
from rag.embeddings import get_embeddings
from rag.llm import get_llm
from rag.vector_store import load_vector_store


class AIAssistant:
    """AI assistant with optional RAG grounding over your documents."""

    def __init__(self, use_rag: bool = True):
        self.llm = get_llm()
        self.memory = get_memory()
        self.use_rag = use_rag
        self._rag_chain = None

        if use_rag and VECTOR_DB_DIR.exists():
            embeddings = get_embeddings()
            vector_store = load_vector_store(VECTOR_DB_DIR, embeddings)
            self._rag_chain = build_rag_chain(vector_store)

    def ask(self, question: str) -> str:
        if self._rag_chain:
            result = self._rag_chain.invoke({"query": question})
            answer = result["result"]
        else:
            history = self.memory.load_memory_variables({})
            prompt = (
                "You are a helpful AI assistant.\n"
                f"History:\n{history.get('history', '')}\n\n"
                f"User: {question}\n"
                "Assistant:"
            )
            answer = self.llm.invoke(prompt)

        self.memory.save_context({"input": question}, {"output": answer})
        return answer
