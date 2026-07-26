"""
Fine-tune Llama 3 on medical instruction data using LoRA / QLoRA (SFT).

Usage:
  cd finetune
  pip install -r requirements.txt
  huggingface-cli login          # required for meta-llama/Meta-Llama-3-8B-Instruct
  python train.py

  # Custom data / output:
  python train.py --data data/medical_sft.jsonl --output-dir output/my-run

Requires:
  - NVIDIA GPU with 8+ GB VRAM (QLoRA) or 16+ GB (LoRA fp16)
  - Hugging Face account with access to Llama 3 weights
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from config import OUTPUT_DIR, TrainConfig
from dataset import build_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Llama 3 on medical SFT data (LoRA)")
    p.add_argument("--base-model", default=None, help="Hugging Face model id")
    p.add_argument("--data", default=None, help="Path to medical_sft.jsonl")
    p.add_argument("--output-dir", default=None, help="Where to save LoRA adapters")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-seq-length", type=int, default=None)
    p.add_argument("--no-4bit", action="store_true", help="Disable QLoRA (full fp16 LoRA)")
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def apply_cli_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    if args.base_model:
        cfg.base_model = args.base_model
    if args.data:
        cfg.data_path = args.data
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.epochs is not None:
        cfg.num_train_epochs = args.epochs
    if args.batch_size is not None:
        cfg.per_device_train_batch_size = args.batch_size
        cfg.per_device_eval_batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.max_seq_length is not None:
        cfg.max_seq_length = args.max_seq_length
    if args.no_4bit:
        cfg.use_4bit = False
    if args.lora_r is not None:
        cfg.lora_r = args.lora_r
    if args.seed is not None:
        cfg.seed = args.seed
    return cfg


def check_environment(cfg: TrainConfig) -> None:
    if not torch.cuda.is_available():
        print(
            "WARNING: No CUDA GPU detected. Llama 3 fine-tuning needs a GPU.\n"
            "         Training will likely fail or be extremely slow on CPU."
        )
    if cfg.use_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            print(
                "ERROR: bitsandbytes is required for QLoRA (--no-4bit to use fp16 LoRA instead)."
            )
            sys.exit(1)


def load_model_and_tokenizer(cfg: TrainConfig):
    print(f"\n>>> Loading EXISTING pretrained model from Hugging Face...")
    print(f"    Model: {cfg.base_model}")
    print(f"    (8 billion parameters — already trained by Meta, NOT from scratch)\n")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = None
    if cfg.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if not cfg.use_4bit else None,
        trust_remote_code=cfg.trust_remote_code,
    )

    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print("\n>>> Fine-tuning setup:")
    print("    Base Llama 3 weights: FROZEN (kept as pretrained)")
    print("    LoRA adapters:        TRAINABLE (learn medical Q&A)")
    model.print_trainable_parameters()
    return model, tokenizer


def train(cfg: TrainConfig) -> Path:
    cfg.ensure_dirs()
    check_environment(cfg)

    print(f"Base model : {cfg.base_model}")
    print(f"Data       : {cfg.data_path}")
    print(f"Output     : {cfg.output_dir}")
    print(f"QLoRA 4bit : {cfg.use_4bit}")

    model, tokenizer = load_model_and_tokenizer(cfg)

    dataset = build_dataset(cfg.data_path, tokenizer, val_split=cfg.val_split, seed=cfg.seed)
    print(f"Train rows : {len(dataset['train'])}")
    print(f"Val rows   : {len(dataset['validation'])}")

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        report_to="none",
        seed=cfg.seed,
        optim="paged_adamw_8bit" if cfg.use_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        max_seq_length=cfg.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )
    sig = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    print("\nStarting fine-tuning...\n")
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    run_info = {
        "base_model": cfg.base_model,
        "data_path": cfg.data_path,
        "method": "QLoRA" if cfg.use_4bit else "LoRA",
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "epochs": cfg.num_train_epochs,
        "learning_rate": cfg.learning_rate,
        "max_seq_length": cfg.max_seq_length,
        "train_samples": len(dataset["train"]),
        "val_samples": len(dataset["validation"]),
    }
    info_path = Path(cfg.output_dir) / "run_info.json"
    info_path.write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    print(f"\nFine-tuning complete!")
    print(f"  LoRA adapters : {cfg.output_dir}")
    print(f"  Run metadata  : {info_path}")
    print(f"\nNext: python inference.py --adapter-dir {cfg.output_dir}")
    return Path(cfg.output_dir)


def main() -> None:
    cfg = apply_cli_overrides(TrainConfig(), parse_args())
    train(cfg)


if __name__ == "__main__":
    main()
