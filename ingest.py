from config.settings import DATA_RAW_DIR, VECTOR_DB_DIR
from rag.embeddings import get_embeddings
from rag.loader import load_documents
from rag.splitter import split_documents
from rag.utils import setup_logging
from rag.vector_store import build_vector_store


def main():
    setup_logging()
    docs = load_documents(DATA_RAW_DIR)
    if not docs:
        raise SystemExit(f"No documents found in {DATA_RAW_DIR}")

    chunks = split_documents(docs)
    embeddings = get_embeddings()
    build_vector_store(chunks, embeddings, VECTOR_DB_DIR)
    print(f"Indexed {len(chunks)} chunks into {VECTOR_DB_DIR}")


if __name__ == "__main__":
    main()
