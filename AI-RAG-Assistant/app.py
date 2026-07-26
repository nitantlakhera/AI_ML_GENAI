import streamlit as st

from config.settings import VECTOR_DB_DIR
from rag.chain import build_rag_chain
from rag.embeddings import get_embeddings
from rag.vector_store import load_vector_store


@st.cache_resource
def load_chain():
    embeddings = get_embeddings()
    vector_store = load_vector_store(VECTOR_DB_DIR, embeddings)
    return build_rag_chain(vector_store)


st.set_page_config(page_title="AI RAG Assistant", page_icon="🤖")
st.title("AI RAG Assistant")
st.caption("Ask questions grounded in your documents.")

try:
    chain = load_chain()
except Exception as exc:
    st.error(
        "Could not load the vector database. Run `uv run python ingest.py` first."
    )
    st.stop()

question = st.text_input("Ask a question:")
if question:
    with st.spinner("Thinking..."):
        result = chain.invoke({"query": question})

    st.subheader("Answer")
    st.write(result["result"])

    with st.expander("Sources"):
        for doc in result.get("source_documents", []):
            source = doc.metadata.get("source", "unknown")
            st.write(f"- {source}")
