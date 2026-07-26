# Llama 3 Medical Fine-Tuning

Fine-tune an **existing, pretrained Meta Llama 3 8B Instruct** model on medical Q&A.

> **📖 START HERE → [COMPLETE_MANUAL.md](COMPLETE_MANUAL.md)**  
> Single file with **everything**: user guide, architecture, all diagrams, commands, config, troubleshooting.

```
Existing Llama 3 (pretrained)  +  Medical Q&A data  →  Fine-tuned medical Llama 3
     8B params (frozen)              your JSONL           LoRA adapter (~13M params)
```

> **Disclaimer:** For **education and research only** — not for clinical use.

---

## Quick start

```bash
cd finetune
pip install -r requirements.txt
huggingface-cli login
python setup_check.py
python train.py
python inference.py
```

---

## All documentation

| Document | Contents |
|----------|----------|
| **[COMPLETE_MANUAL.md](COMPLETE_MANUAL.md)** | **Master doc — user guide + architecture + all 14 diagrams + commands** |
| [USER_GUIDE.md](USER_GUIDE.md) | Step-by-step setup (also in manual) |
| [FLOW.md](FLOW.md) | Flow diagrams (also in manual) |
| [ARCHITECTURE.md](ARCHITECTURE.md) | LoRA/QLoRA architecture (also in manual) |
| [diagrams/](diagrams/) | PNG exports — run `python render_diagrams.py` |

---

## Folder structure

```
finetune/
├── COMPLETE_MANUAL.md     ← ALL docs, diagrams, user manual (start here)
├── train.py               ← fine-tune existing Llama 3
├── inference.py           ← chat with fine-tuned model
├── setup_check.py         ← verify setup before training
├── data/medical_sft.jsonl ← sample medical Q&A (20 pairs)
└── output/                ← fine-tuned adapters saved here
```

See [COMPLETE_MANUAL.md](COMPLETE_MANUAL.md) for full details.
