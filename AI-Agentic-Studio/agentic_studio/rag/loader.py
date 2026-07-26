"""Document loading with rich metadata.

Metadata captured here (source, filetype, page, title, mtime) is what makes
metadata filtering and citation-quality answers possible downstream.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agentic_studio.core.types import Document
from agentic_studio.observability.logs import get_logger

logger = get_logger("rag.loader")

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".log"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf", ".json", ".jsonl", ".csv", ".html", ".htm"}


def load_documents(
    root: Path,
    suffixes: Iterable[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> list[Document]:
    """Recursively load every supported file under `root`."""
    root = Path(root)
    if not root.exists():
        logger.warning("source directory does not exist: %s", root)
        return []

    allowed = {s.lower() for s in (suffixes or SUPPORTED_SUFFIXES)}
    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        try:
            documents.extend(load_file(path, extra_metadata))
        except Exception as exc:
            logger.warning("failed to load %s: %s", path, exc)
    logger.info("loaded %d document(s) from %s", len(documents), root)
    return documents


def load_file(path: Path, extra_metadata: dict[str, Any] | None = None) -> list[Document]:
    path = Path(path)
    suffix = path.suffix.lower()
    base = _base_metadata(path)
    if extra_metadata:
        base.update(extra_metadata)

    if suffix == ".pdf":
        return _load_pdf(path, base)
    if suffix in {".json", ".jsonl"}:
        return _load_json(path, base)
    if suffix == ".csv":
        return _load_csv(path, base)
    if suffix in {".html", ".htm"}:
        return _load_html(path, base)
    return _load_text(path, base)


def load_texts(texts: list[str], source: str = "inline") -> list[Document]:
    """Wrap raw strings as documents; handy for tests and API ingestion."""
    return [
        Document(text=text, metadata={"source": f"{source}#{index}", "filetype": "text"})
        for index, text in enumerate(texts)
    ]


def _base_metadata(path: Path) -> dict[str, Any]:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return {
        "source": str(path),
        "filename": path.name,
        "filetype": path.suffix.lower().lstrip("."),
        "modified_at": mtime,
    }


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


def _load_text(path: Path, metadata: dict[str, Any]) -> list[Document]:
    text = _read_text(path)
    title = _markdown_title(text) or path.stem
    return [Document(text=text, metadata={**metadata, "title": title})]


def _markdown_title(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
        if stripped:
            break
    return None


def _load_pdf(path: Path, metadata: dict[str, Any]) -> list[Document]:
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("pypdf not installed; skipping %s", path)
        return []

    reader = PdfReader(str(path))
    info = getattr(reader, "metadata", None)
    title = (getattr(info, "title", None) or path.stem) if info else path.stem

    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        documents.append(
            Document(
                text=text,
                metadata={**metadata, "title": title, "page": page_number,
                          "total_pages": len(reader.pages)},
            )
        )
    return documents


def _load_json(path: Path, metadata: dict[str, Any]) -> list[Document]:
    raw = _read_text(path)
    documents: list[Document] = []

    if path.suffix.lower() == ".jsonl":
        records = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    else:
        parsed = json.loads(raw)
        records = parsed if isinstance(parsed, list) else [parsed]

    for index, record in enumerate(records):
        text = _flatten(record)
        if text.strip():
            documents.append(
                Document(text=text, metadata={**metadata, "record": index, "title": path.stem})
            )
    return documents


def _load_csv(path: Path, metadata: dict[str, Any]) -> list[Document]:
    raw = _read_text(path)
    reader = csv.DictReader(io.StringIO(raw))
    documents: list[Document] = []
    for index, row in enumerate(reader):
        text = "\n".join(f"{key}: {value}" for key, value in row.items() if value)
        if text.strip():
            documents.append(
                Document(text=text, metadata={**metadata, "row": index, "title": path.stem})
            )
    return documents


def _load_html(path: Path, metadata: dict[str, Any]) -> list[Document]:
    import re as _re

    raw = _read_text(path)
    title_match = _re.search(r"<title>(.*?)</title>", raw, _re.S | _re.I)
    body = _re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=_re.S | _re.I)
    body = _re.sub(r"<[^>]+>", " ", body)
    body = _re.sub(r"\s+", " ", body).strip()
    title = title_match.group(1).strip() if title_match else path.stem
    return [Document(text=body, metadata={**metadata, "title": title})] if body else []


def _flatten(value: Any, prefix: str = "") -> str:
    if isinstance(value, dict):
        return "\n".join(_flatten(v, f"{prefix}{k}.") for k, v in value.items())
    if isinstance(value, list):
        return "\n".join(_flatten(v, prefix) for v in value)
    return f"{prefix.rstrip('.')}: {value}" if prefix else str(value)
