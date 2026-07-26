# Fine-Tune — Diagram Images (PNG)

Rendered from Mermaid diagrams in **`COMPLETE_MANUAL.md`** (master doc) and the individual doc files.

## Regenerate all PNGs

```bash
cd finetune
python render_diagrams.py
```

## Source files

| Source | Diagrams |
|--------|----------|
| **`COMPLETE_MANUAL.md`** | All 14 diagrams (master manual) |
| `FLOW.md` | 9 flow diagrams |
| `ARCHITECTURE.md` | 5 architecture diagrams |
| `USER_GUIDE.md` | 2 setup/training diagrams |

## Diagram index (from COMPLETE_MANUAL.md)

| # | Topic | Section |
|---|-------|---------|
| 1 | Big picture (train once, use many) | §22 |
| 2 | File dependency graph | §23 |
| 3 | Training flow | §24 |
| 4 | One training step (sequence) | §25 |
| 5 | Inference flow | §26 |
| 6 | Data pipeline (JSONL → template) | §27 |
| 7 | LoRA vs full fine-tuning vs QLoRA | §28 |
| 8 | End-to-end lifecycle | §29 |
| 9 | MiniGPT vs Llama 3 fine-tune | §30 |
| 10 | LoRA math (W + B×A) | §31 |
| 11 | QLoRA GPU memory layout | §32 |
| 12 | Full Llama 3 + LoRA stack | §33 |
| 13 | SFT overview | §34 |
| 14 | Training sequence diagram | §35 |

PNG files are named: `{SOURCE_FILE}__{NN}__{slug}.png`

Example: `COMPLETE_MANUAL__22__flowchart-lr.png`
