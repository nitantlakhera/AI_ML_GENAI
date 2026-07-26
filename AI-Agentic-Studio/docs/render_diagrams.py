"""
render_diagrams.py  (AI-Agentic-Studio/docs)
---------------------------------------------
Render every ```mermaid block in docs/*.md to PNG files in diagrams/
via the mermaid.ink service.

    cd AI-Agentic-Studio/docs
    python render_diagrams.py
"""

import base64
import json
import re
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
OUT_DIR = DOCS_DIR / "diagrams"
MERMAID_BLOCK = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)

DOC_FILES = [
    "ARCHITECTURE.md",
    "LEARNING-GUIDE.md",
    "LEARNING-PATH.md",
    "GAP-ANALYSIS.md",
    "CONCEPTS.md",
]

# Friendly PNG names for key learner / architecture diagrams
ALIASES: dict[tuple[str, int], str] = {
    ("ARCHITECTURE", 1): "architecture-high-level-system.png",
    ("ARCHITECTURE", 2): "architecture-rag-pipeline.png",
    ("ARCHITECTURE", 3): "architecture-llm-router.png",
    ("ARCHITECTURE", 4): "architecture-stategraph.png",
    ("ARCHITECTURE", 5): "architecture-agent-modes.png",
    ("ARCHITECTURE", 6): "architecture-tool-execution.png",
    ("ARCHITECTURE", 7): "architecture-guardrail-boundaries.png",
    ("ARCHITECTURE", 8): "architecture-api.png",
    ("ARCHITECTURE", 9): "architecture-rag-query-sequence.png",
    ("LEARNING-GUIDE", 1): "learning-three-interfaces.png",
    ("LEARNING-GUIDE", 2): "flow-a-studio-ask.png",
    ("LEARNING-GUIDE", 3): "flow-b-studio-ingest.png",
    ("LEARNING-GUIDE", 4): "flow-c-studio-agent.png",
    ("LEARNING-GUIDE", 5): "flow-d-studio-ui-chat.png",
    ("LEARNING-GUIDE", 6): "learning-four-layer-architecture.png",
}


def encode_mermaid(graph: str) -> str:
    state = {"code": graph, "mermaid": {"theme": "default"}}
    raw = json.dumps(state).encode("utf-8")
    return "pako:" + base64.urlsafe_b64encode(zlib.compress(raw, level=9)).decode("ascii")


def render(graph: str, out_path: Path) -> bool:
    url = f"https://mermaid.ink/img/{encode_mermaid(graph)}?type=png&bgColor=white"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            out_path.write_bytes(resp.read())
        return True
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode(errors="replace")[:300]
        except Exception:
            pass
        print(f"  ! HTTP {e.code} for {out_path.name}: {body}")
        return False
    except Exception as e:
        print(f"  ! Failed {out_path.name}: {e}")
        return False


def slug(text: str, max_len: int = 40) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text[:max_len] or "diagram"


def png_name(stem: str, index: int, graph: str) -> str:
    alias = ALIASES.get((stem, index))
    if alias:
        return alias
    first_line = next((ln for ln in graph.splitlines() if ln.strip()), "diagram")
    return f"{stem}__{index:02d}__{slug(first_line)}.png"


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    total, ok = 0, 0
    for doc in DOC_FILES:
        path = DOCS_DIR / doc
        if not path.exists():
            print(f"Skip (not found): {doc}")
            continue
        blocks = MERMAID_BLOCK.findall(path.read_text(encoding="utf-8"))
        print(f"{doc}: {len(blocks)} diagram(s)")
        for i, graph in enumerate(blocks, start=1):
            graph = graph.strip()
            name = png_name(path.stem, i, graph)
            total += 1
            if render(graph, OUT_DIR / name):
                ok += 1
                print(f"  OK  {name}")
            time.sleep(0.5)
    print(f"\nDone: {ok}/{total} PNGs written to {OUT_DIR}")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
