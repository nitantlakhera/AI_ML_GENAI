"""
Merge LoRA adapters into the base model for standalone deployment.

Usage:
  python merge_lora.py --adapter-dir output/llama3-medical-lora --merged-dir output/llama3-medical-merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser(description="Merge LoRA weights into base Llama 3")
    p.add_argument("--base-model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--merged-dir", required=True)
    args = p.parse_args()

    adapter = Path(args.adapter_dir)
    merged = Path(args.merged_dir)
    merged.mkdir(parents=True, exist_ok=True)

    print(f"Loading base: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(base, str(adapter))
    print("Merging LoRA into base weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {merged}")
    model.save_pretrained(merged)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter))
    tokenizer.save_pretrained(merged)
    print("Done.")


if __name__ == "__main__":
    main()
