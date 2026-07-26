"""
Generate PDF documentation from markdown sources and diagram PNGs.

Output: docs/pdf/
  - user-manual.pdf
  - architecture.pdf
  - flow-diagrams.pdf
  - complete-guide.pdf (combined essentials)

Usage:
    uv run python scripts/generate_pdfs.py
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from fpdf import FPDF
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
IMAGES = DOCS / "images"
PDF_DIR = DOCS / "pdf"


def to_pdf_text(text: str) -> str:
    """Normalize unicode for Helvetica (latin-1) PDF output."""
    replacements = {
        "\u2014": "-",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "*",
        "\u2192": "->",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("latin-1", errors="replace").decode("latin-1")


class DocPDF(FPDF):
    def __init__(self, title: str):
        super().__init__()
        self.doc_title = to_pdf_text(title)

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, self.doc_title, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def write_line(self, text: str, font: str = "Helvetica", style: str = "", size: int = 10):
        self.set_font(font, style, size)
        self.multi_cell(0, 5, to_pdf_text(text))


def clean_inline(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("**", "").replace("`", "")
    return to_pdf_text(text.strip())


def render_markdown(pdf: DocPDF, md_path: Path, base_dir: Path) -> None:
    if not md_path.exists():
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(pdf.epw, 5, to_pdf_text(f"[Missing file: {md_path.name}]"))
        return

    w = pdf.epw
    in_code = False
    for raw_line in md_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()

        if line.startswith("```"):
            in_code = not in_code
            if in_code:
                pdf.ln(2)
            continue

        if in_code:
            pdf.set_font("Courier", "", 8)
            pdf.set_text_color(20, 20, 20)
            pdf.multi_cell(pdf.epw, 4, to_pdf_text(line))
            continue

        if not line.strip():
            pdf.ln(3)
            continue

        if line.startswith("# "):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 16)
            pdf.set_text_color(0, 51, 102)
            pdf.multi_cell(w, 8, clean_inline(line[2:]))
            pdf.ln(2)
            continue

        if line.startswith("## "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(0, 80, 140)
            pdf.multi_cell(w, 7, clean_inline(line[3:]))
            pdf.ln(1)
            continue

        if line.startswith("### "):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(w, 6, clean_inline(line[4:]))
            continue

        if line.startswith("---"):
            pdf.ln(2)
            continue

        img_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if img_match:
            alt, rel = img_match.group(1), img_match.group(2)
            img_path = (base_dir / rel).resolve()
            if img_path.exists():
                add_image_page(pdf, img_path, alt or img_path.stem)
            continue

        if line.startswith("|"):
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(w, 5, clean_inline(line.replace("|", "  ").strip()))
            continue

        if line.startswith("- ") or line.startswith("* "):
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(w, 5, "  * " + clean_inline(line[2:]))
            continue

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(w, 5, clean_inline(line))


def add_image_page(pdf: DocPDF, image_path: Path, title: str) -> None:
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(0, 51, 102)
    pdf.multi_cell(pdf.epw, 8, to_pdf_text(title))
    pdf.ln(4)

    with Image.open(image_path) as img:
        w_px, h_px = img.size
    page_w = pdf.w - 20
    max_h = pdf.h - 50
    ratio = w_px / h_px
    disp_w = page_w
    disp_h = disp_w / ratio
    if disp_h > max_h:
        disp_h = max_h
        disp_w = disp_h * ratio

    x = (pdf.w - disp_w) / 2
    pdf.image(str(image_path), x=x, y=pdf.get_y(), w=disp_w, h=disp_h)


def cover_page(pdf: DocPDF, title: str, subtitle: str) -> None:
    pdf.add_page()
    pdf.ln(40)
    w = pdf.epw
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(0, 51, 102)
    pdf.multi_cell(w, 12, to_pdf_text(title), align="C")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(w, 8, to_pdf_text(subtitle), align="C")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(w, 6, to_pdf_text(f"Project: {ROOT.name}"), align="C")
    pdf.multi_cell(w, 6, "Generated by scripts/generate_pdfs.py", align="C")


def build_user_manual() -> Path:
    pdf = DocPDF("AI ML GenAI - User Manual")
    cover_page(
        pdf,
        "AI / ML / GenAI",
        "User Manual - How to Run and Use",
    )
    pdf.add_page()
    render_markdown(pdf, DOCS / "getting-started.md", DOCS)
    pdf.add_page()
    render_markdown(pdf, DOCS / "run-and-dependencies.md", DOCS)
    pdf.add_page()
    render_markdown(pdf, DOCS / "user-manual.md", DOCS)
    pdf.add_page()
    render_markdown(pdf, DOCS / "health-check-and-setup.md", DOCS)
    pdf.add_page()
    render_markdown(pdf, DOCS / "api.md", DOCS)
    pdf.add_page()
    render_markdown(pdf, DOCS / "minigpt.md", DOCS)

    out = PDF_DIR / "user-manual.pdf"
    pdf.output(str(out))
    return out


def build_architecture() -> Path:
    pdf = DocPDF("AI ML GenAI - Architecture")
    cover_page(pdf, "System Architecture", "Modules, layers, and data flow")
    pdf.add_page()
    render_markdown(pdf, DOCS / "architecture.md", DOCS)

    overview = IMAGES / "architecture-overview.png"
    if overview.exists():
        add_image_page(pdf, overview, "High-Level System Architecture")

    out = PDF_DIR / "architecture.pdf"
    pdf.output(str(out))
    return out


def build_flow_diagrams() -> Path:
    pdf = DocPDF("AI ML GenAI - Flow Diagrams")
    cover_page(pdf, "Flow Diagrams", "RAG, Agent, MCP, and Application Modes")

    diagrams = [
        ("architecture-overview.png", "System Architecture Overview"),
        ("rag-flow.png", "RAG Pipeline Flow"),
        ("agent-flow.png", "AI Agent Flow"),
        ("mcp-flow.png", "MCP (Model Context Protocol) Flow"),
        ("app-modes-flow.png", "Streamlit Application Modes"),
    ]
    for filename, title in diagrams:
        path = IMAGES / filename
        if path.exists():
            add_image_page(pdf, path, title)

    pdf.add_page()
    render_markdown(pdf, DOCS / "flow-diagrams.md", DOCS)

    out = PDF_DIR / "flow-diagrams.pdf"
    pdf.output(str(out))
    return out


def build_complete_guide() -> Path:
    pdf = DocPDF("AI ML GenAI - Complete Guide")
    cover_page(
        pdf,
        "Complete Guide",
        "Setup, Usage, Architecture & Diagrams",
    )

    for md_file in [
        "getting-started.md",
        "run-and-dependencies.md",
        "user-manual.md",
        "architecture.md",
        "flow-diagrams.md",
        "models.md",
        "health-check-and-setup.md",
    ]:
        pdf.add_page()
        render_markdown(pdf, DOCS / md_file, DOCS)

    for filename, title in [
        ("architecture-overview.png", "Architecture Overview"),
        ("rag-flow.png", "RAG Flow"),
        ("agent-flow.png", "Agent Flow"),
        ("mcp-flow.png", "MCP Flow"),
        ("app-modes-flow.png", "App Modes"),
    ]:
        path = IMAGES / filename
        if path.exists():
            add_image_page(pdf, path, title)

    out = PDF_DIR / "complete-guide.pdf"
    pdf.output(str(out))
    return out


def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        build_user_manual(),
        build_architecture(),
        build_flow_diagrams(),
        build_complete_guide(),
    ]
    for path in outputs:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"Created: {path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
