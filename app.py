import streamlit as st

from agents.executor import run_agent
from chat.assistant import AIAssistant
from chat.bot import ChatBot
from config.settings import VECTOR_DB_DIR
from rag.chain import build_rag_chain
from rag.embeddings import get_embeddings
from rag.vector_store import load_vector_store

st.set_page_config(page_title="AI ML GenAI", page_icon="🤖", layout="wide")
st.title("AI / ML / GenAI Workspace")
st.caption("Chatbot · AI Assistant · RAG · Agents")

mode = st.sidebar.selectbox(
    "Mode",
    ["Chatbot", "AI Assistant", "RAG Q&A", "Agent"],
)

if mode == "Chatbot":
    bot = ChatBot()
    question = st.text_input("Message:")
    if question:
        st.write(bot.chat(question))

elif mode == "AI Assistant":
    use_rag = st.sidebar.checkbox("Use RAG (if index exists)", value=VECTOR_DB_DIR.exists())
    assistant = AIAssistant(use_rag=use_rag)
    question = st.text_input("Ask the assistant:")
    if question:
        st.write(assistant.ask(question))

elif mode == "RAG Q&A":
    @st.cache_resource
    def load_chain():
        embeddings = get_embeddings()
        vector_store = load_vector_store(VECTOR_DB_DIR, embeddings)
        return build_rag_chain(vector_store)

    try:
        chain = load_chain()
    except Exception:
        st.error("Run `uv run python ingest.py` first to build the vector index.")
        st.stop()

    question = st.text_input("Question (grounded in your documents):")
    if question:
        with st.spinner("Thinking..."):
            result = chain.invoke({"query": question})
        st.subheader("Answer")
        st.write(result["result"])
        with st.expander("Sources"):
            for doc in result.get("source_documents", []):
                st.write(doc.metadata.get("source", "unknown"))

elif mode == "Agent":
    st.info("Agents work best with API LLMs (`USE_API_LLM=true` in `.env`).")
    question = st.text_input("Agent task:")
    if question:
        with st.spinner("Agent working..."):
            try:
                st.write(run_agent(question))
            except Exception as exc:
                st.error(f"Agent error: {exc}")
