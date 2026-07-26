"""
Chat with a fine-tuned Llama 3 medical LoRA adapter.

Usage:
  python inference.py
  python inference.py --adapter-dir output/llama3-medical-lora
  python inference.py --prompt "What is hypertension?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import OUTPUT_DIR


DEFAULT_ADAPTER = OUTPUT_DIR / "llama3-medical-lora"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference with fine-tuned Llama 3 medical LoRA")
    p.add_argument("--base-model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--adapter-dir", default=str(DEFAULT_ADAPTER))
    p.add_argument("--prompt", default=None, help="Single question (non-interactive)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--no-4bit", action="store_true")
    return p.parse_args()


def build_messages(question: str) -> list[dict]:
    return [{"role": "user", "content": question}]


def load_model(base_model: str, adapter_dir: str, use_4bit: bool):
    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        print(f"ERROR: Adapter not found at {adapter_path}")
        print("Run training first:  python train.py")
        sys.exit(1)

    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if not use_4bit else None,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def generate(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    messages = build_messages(question)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def chat_loop(model, tokenizer, args) -> None:
    print("=" * 60)
    print("  Llama 3 Medical Fine-Tuned Chat")
    print("  Type 'quit' or 'exit' to stop.")
    print("  DISCLAIMER: For education only — not medical advice.")
    print("=" * 60)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Bye!")
            break
        answer = generate(
            model, tokenizer, question,
            args.max_new_tokens, args.temperature, args.top_p,
        )
        print(f"\nAssistant: {answer}")


def main() -> None:
    args = parse_args()
    use_4bit = not args.no_4bit

    print(f"Loading base model : {args.base_model}")
    print(f"Loading LoRA adapter: {args.adapter_dir}")
    model, tokenizer = load_model(args.base_model, args.adapter_dir, use_4bit)

    if args.prompt:
        answer = generate(
            model, tokenizer, args.prompt,
            args.max_new_tokens, args.temperature, args.top_p,
        )
        print(answer)
    else:
        chat_loop(model, tokenizer, args)


if __name__ == "__main__":
    main()
