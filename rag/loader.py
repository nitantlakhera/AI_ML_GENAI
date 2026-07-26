from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_documents(raw_dir: Path):
    """Load PDF, TXT, and Markdown files from the raw data directory."""
    docs = []

    for pdf in sorted(raw_dir.glob("*.pdf")):
        docs.extend(PyPDFLoader(str(pdf)).load())

    for txt in sorted(raw_dir.glob("*.txt")):
        docs.extend(TextLoader(str(txt), encoding="utf-8").load())

    for md in sorted(raw_dir.glob("*.md")):
        docs.extend(TextLoader(str(md), encoding="utf-8").load())

    return docs
