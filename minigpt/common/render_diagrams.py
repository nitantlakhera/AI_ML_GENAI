"""
render_diagrams.py  (common/ docs)
----------------------------------
Render every ```mermaid block in this folder's framework-agnostic docs to PNG
(in `diagrams/`), via the mermaid.ink service.

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

DOC_FILES = ["COMPARISON.md", "HOW_TO_BUILD_AN_LLM.md"]  # Docs that contain mermaid
OUT_DIR = Path("diagrams")
MERMAID_BLOCK = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)


def encode_mermaid(graph: str) -> str:
    state = {"code": graph, "mermaid": {"theme": "default"}}
    raw = json.dumps(state).encode("utf-8")
    return "pako:" + base64.urlsafe_b64encode(zlib.compress(raw, level=9)).decode("ascii")


def render(graph: str, out_path: Path) -> bool:
    url = f"https://mermaid.ink/img/{encode_mermaid(graph)}?type=png&bgColor=white"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
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


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    total, ok = 0, 0
    for doc in DOC_FILES:
        path = Path(doc)
        if not path.exists():
            print(f"Skip (not found): {doc}")
            continue
        blocks = MERMAID_BLOCK.findall(path.read_text(encoding="utf-8"))
        print(f"{doc}: {len(blocks)} diagram(s)")
        for i, graph in enumerate(blocks, start=1):
            graph = graph.strip()
            first_line = next((ln for ln in graph.splitlines() if ln.strip()), "diagram")
            name = f"{path.stem}__{i:02d}__{slug(first_line)}.png"
            total += 1
            if render(graph, OUT_DIR / name):
                ok += 1
                print(f"  OK  {name}")
            time.sleep(0.5)
    print(f"\nDone: {ok}/{total} PNGs written to {OUT_DIR.resolve()}")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
