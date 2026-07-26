"""
Validate a medical SFT JSONL file before training.

Usage:
  python validate_data.py
  python validate_data.py --data data/my_data.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate(path: str) -> bool:
    p = Path(path)
    print(f"Validating: {p}")
    if not p.exists():
        print(f"  FAIL — file not found")
        return False

    errors = 0
    rows = 0
    with p.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  FAIL line {line_no}: invalid JSON — {e}")
                errors += 1
                continue
            if "instruction" not in row:
                print(f"  FAIL line {line_no}: missing 'instruction'")
                errors += 1
            if "output" not in row:
                print(f"  FAIL line {line_no}: missing 'output'")
                errors += 1
            if not row.get("instruction", "").strip():
                print(f"  FAIL line {line_no}: empty 'instruction'")
                errors += 1
            if not row.get("output", "").strip():
                print(f"  FAIL line {line_no}: empty 'output'")
                errors += 1

    if rows == 0:
        print("  FAIL — no data rows found")
        return False

    if errors:
        print(f"\n  Result: FAIL ({errors} error(s) in {rows} rows)")
        return False

    print(f"  OK — {rows} valid row(s)")
    try:
        loaded = load_jsonl(path)
        print(f"  Sample instruction: {loaded[0]['instruction'][:60]}...")
    except Exception as e:
        print(f"  WARN — load_jsonl raised: {e}")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Validate medical SFT JSONL")
    p.add_argument("--data", default="data/medical_sft.jsonl")
    args = p.parse_args()
    ok = validate(args.data)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
