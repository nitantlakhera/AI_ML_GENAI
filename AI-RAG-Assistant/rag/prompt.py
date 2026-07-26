from langchain_core.prompts import PromptTemplate

RAG_PROMPT = PromptTemplate(
    template=(
        "You are a helpful AI assistant. Use ONLY the context below to answer.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    input_variables=["context", "question"],
)
