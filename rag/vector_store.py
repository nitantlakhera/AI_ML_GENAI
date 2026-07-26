from pathlib import Path

from langchain_community.vectorstores import FAISS


def build_vector_store(chunks, embeddings, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(save_dir))
    return vector_store


def load_vector_store(save_dir: Path, embeddings):
    return FAISS.load_local(
        str(save_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
