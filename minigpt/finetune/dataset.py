"""
Medical SFT dataset loader and Llama 3 chat-template formatter.

Expected JSONL format (one object per line):
  {"instruction": "...", "input": "...", "output": "..."}

`input` is optional (use "" when not needed).
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, DatasetDict


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "instruction" not in row or "output" not in row:
                raise ValueError(
                    f"{path}:{line_no} — each row needs 'instruction' and 'output'"
                )
            rows.append(
                {
                    "instruction": row["instruction"].strip(),
                    "input": row.get("input", "").strip(),
                    "output": row["output"].strip(),
                }
            )

    if not rows:
        raise ValueError(f"No training rows found in {path}")
    return rows


def row_to_messages(example: dict) -> list[dict]:
    """Convert one SFT row to Hugging Face chat messages."""
    user_content = example["instruction"]
    if example.get("input"):
        user_content = f"{user_content}\n\n{example['input']}"
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output"]},
    ]


def format_with_tokenizer(example: dict, tokenizer) -> dict:
    """Apply the model's built-in chat template (Llama 3 Instruct)."""
    messages = row_to_messages(example)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def build_dataset(
    data_path: str | Path,
    tokenizer,
    val_split: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    rows = load_jsonl(data_path)
    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=val_split, seed=seed)

    def _map(batch):
        return format_with_tokenizer(batch, tokenizer)

    return DatasetDict(
        {
            "train": split["train"].map(_map, remove_columns=split["train"].column_names),
            "validation": split["test"].map(_map, remove_columns=split["test"].column_names),
        }
    )
