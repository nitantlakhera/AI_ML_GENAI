from langchain.chains import RetrievalQA

from rag.llm import get_llm
from rag.prompt import RAG_PROMPT
from rag.retriever import get_retriever


def build_rag_chain(vector_store):
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=get_retriever(vector_store),
        chain_type_kwargs={"prompt": RAG_PROMPT},
        return_source_documents=True,
    )
