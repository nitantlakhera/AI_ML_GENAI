from langchain_core.documents import Document

from rag.splitter import split_documents


def test_splitter_creates_multiple_chunks():
    docs = [Document(page_content="word " * 600)]
    chunks = split_documents(docs)
    assert len(chunks) > 1
