"""
render_diagrams.py
------------------
Extract every ```mermaid code block from the project's Markdown docs and
render each one to a PNG image in the `diagrams/` folder.

Rendering is done via the free mermaid.ink service (requires internet).
Each PNG is named:  <source-doc>__<NN>__<first-title-line>.png
"""

import base64
import re
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

# Markdown files that contain mermaid diagrams.
DOC_FILES = [
    "NEURAL_NETWORK.md",
    "FLOW.md",
    "DOCKER.md",
]

OUT_DIR = Path("diagrams")                        # Where PNGs are written
MERMAID_BLOCK = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)  # Match fenced mermaid blocks


def encode_mermaid(graph: str) -> str:
    """Encode a mermaid graph the same way mermaid.live does (pako/zlib + base64url)."""
    state = {"code": graph, "mermaid": {"theme": "default"}}  # Minimal state object
    import json                                    # Local import keeps top clean
    raw = json.dumps(state).encode("utf-8")        # JSON -> bytes
    compressed = zlib.compress(raw, level=9)       # zlib deflate (matches pako.deflate)
    b64 = base64.urlsafe_b64encode(compressed).decode("ascii")  # base64url
    return f"pako:{b64}"                            # mermaid.ink pako prefix


def render(graph: str, out_path: Path) -> bool:
    """Download a single PNG for one mermaid graph. Returns True on success."""
    encoded = encode_mermaid(graph)                # Encode the graph
    url = f"https://mermaid.ink/img/{encoded}?type=png&bgColor=white"  # Image URL
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})  # Avoid 403
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # Fetch the image
            data = resp.read()                     # Read PNG bytes
        out_path.write_bytes(data)                 # Save to disk
        return True                                # Success
    except urllib.error.HTTPError as e:            # Server returned an error status
        body = ""                                  # Try to read the error detail
        try:
            body = e.read().decode(errors="replace")[:300]
        except Exception:
            pass
        print(f"  ! HTTP {e.code} for {out_path.name}: {body}")
        return False
    except Exception as e:                         # Network/timeout/other error
        print(f"  ! Failed {out_path.name}: {e}")
        return False


def slug(text: str, max_len: int = 40) -> str:
    """Turn the first line of a graph into a safe filename fragment."""
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()  # Keep alphanumerics
    return text[:max_len] or "diagram"             # Truncate; fallback name


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)                   # Create diagrams/ if missing
    total, ok = 0, 0                               # Counters

    for doc in DOC_FILES:                          # Loop over each Markdown file
        path = Path(doc)
        if not path.exists():                      # Skip missing files
            print(f"Skip (not found): {doc}")
            continue

        text = path.read_text(encoding="utf-8")    # Read the whole file
        blocks = MERMAID_BLOCK.findall(text)       # Find all mermaid blocks
        stem = path.stem                           # e.g. "NEURAL_NETWORK"
        print(f"{doc}: {len(blocks)} diagram(s)")

        for i, graph in enumerate(blocks, start=1):  # Render each block
            graph = graph.strip()                  # Clean whitespace
            # Use the first meaningful line as a label for the filename.
            first_line = next((ln for ln in graph.splitlines() if ln.strip()), "diagram")
            name = f"{stem}__{i:02d}__{slug(first_line)}.png"  # Build filename
            out_path = OUT_DIR / name
            total += 1
            if render(graph, out_path):            # Try to render
                ok += 1
                print(f"  OK  {name}")
            time.sleep(0.5)                        # Be polite to the service

    print(f"\nDone: {ok}/{total} PNGs written to {OUT_DIR.resolve()}")
    return 0 if ok == total else 1                 # Non-zero exit if any failed


if __name__ == "__main__":
    sys.exit(main())
