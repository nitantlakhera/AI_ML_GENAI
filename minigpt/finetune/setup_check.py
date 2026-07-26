"""
Verify that the existing Llama 3 model can be downloaded from Hugging Face
before starting fine-tuning.

Usage:
  cd finetune
  huggingface-cli login
  python setup_check.py
  python setup_check.py --base-model meta-llama/Meta-Llama-3-8B-Instruct
"""

from __future__ import annotations

import argparse
import sys

from config import TrainConfig


def check(base_model: str) -> bool:
    ok = True
    print("=" * 60)
    print("  Llama 3 Fine-Tuning — Setup Check")
    print("  (uses EXISTING pretrained model — not training from scratch)")
    print("=" * 60)

    # 1. Python packages
    print("\n[1/4] Checking Python packages...")
    for pkg in ("torch", "transformers", "peft", "trl", "datasets", "accelerate"):
        try:
            __import__(pkg)
            print(f"  OK  {pkg}")
        except ImportError:
            print(f"  FAIL  {pkg} — run: pip install -r requirements.txt")
            ok = False

    # 2. GPU
    print("\n[2/4] Checking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  OK  CUDA GPU: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  OK  VRAM: {vram:.1f} GB")
            if vram < 8:
                print("  WARN  Less than 8 GB VRAM — QLoRA may fail; try --batch-size 1")
        else:
            print("  WARN  No CUDA GPU — Llama 3 fine-tuning requires an NVIDIA GPU")
    except Exception as e:
        print(f"  FAIL  {e}")
        ok = False

    # 3. Hugging Face auth
    print("\n[3/4] Checking Hugging Face login...")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"  OK  Logged in as: {user.get('name', 'unknown')}")
    except Exception:
        print("  FAIL  Not logged in — run: huggingface-cli login")
        ok = False

    # 4. Access to existing Llama 3 weights
    print(f"\n[4/4] Checking access to existing model: {base_model}")
    try:
        from huggingface_hub import model_info
        info = model_info(base_model)
        print(f"  OK  Model found on Hugging Face ({info.sha[:12]}...)")
        print(f"  OK  This is a PRETRAINED model — fine-tuning will adapt it to medical data")
    except Exception as e:
        print(f"  FAIL  Cannot access model: {e}")
        print("        1. Accept the license at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct")
        print("        2. Run: huggingface-cli login")
        ok = False

    print("\n" + "=" * 60)
    if ok:
        print("  All checks passed!")
        print("  Next: python train.py")
        print("        (downloads Llama 3 weights, then fine-tunes on medical data)")
    else:
        print("  Fix the issues above before running train.py")
    print("=" * 60)
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description="Verify Llama 3 fine-tuning setup")
    p.add_argument("--base-model", default=TrainConfig().base_model)
    args = p.parse_args()
    sys.exit(0 if check(args.base_model) else 1)


if __name__ == "__main__":
    main()
