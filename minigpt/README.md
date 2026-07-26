# MiniGPT — Build a Tiny GPT From Scratch (PyTorch **and** TensorFlow)

> **Part of the unified `AI_ML_GENAI` workspace** at `C:\MY_SPACE\AI_ML_GENAI\minigpt\`.
> See [docs/minigpt.md](../docs/minigpt.md) for how MiniGPT fits with RAG, agents, API, and Streamlit.

A tiny GPT-style language model trained on *Twinkle Twinkle Little Star*, implemented
from scratch in **both PyTorch and TensorFlow/Keras** so you can learn and compare both.

Both versions are the same architecture: **110,720 parameters**, 2 layers, 4 heads,
64-dim embeddings, 32-token context.

---

## Project structure

```
MINIGPT/
├── common/                     # shared by BOTH versions
│   ├── tokenizer.py            #   word ↔ ID tokenizer (pure Python)
│   ├── data/song.txt           #   training data
│   ├── COMPARISON.md           #   MiniGPT vs ChatGPT (framework-agnostic)
│   ├── PYTORCH_VS_TENSORFLOW.md #  side-by-side framework comparison
│   └── diagrams/               #   MiniGPT-vs-ChatGPT PNGs
│
├── pytorch/                    # PyTorch implementation
│   ├── attention.py, transformer.py, model.py
│   ├── dataset.py, train.py, generate.py, chat.py
│   ├── paths.py                #   locates ../common
│   ├── requirements.txt        #   torch
│   ├── Dockerfile, docker-compose.yml
│   ├── README.md, ARCHITECTURE.md, NEURAL_NETWORK.md, FLOW.md, DESIGN.md, DOCKER.md
│   ├── diagrams/               #   rendered PNGs
│   └── checkpoints/            #   trained model (after training)
│
├── finetune/                   # Llama 3 medical fine-tuning (QLoRA / LoRA SFT)
│   ├── COMPLETE_MANUAL.md      #   ALL docs, diagrams, user manual (start here)
│   ├── train.py, inference.py, dataset.py, config.py
│   ├── data/medical_sft.jsonl  #   sample medical Q&A dataset
│   ├── output/                 #   fine-tuned LoRA adapters (after training)
│   ├── README.md, USER_GUIDE.md, FLOW.md, ARCHITECTURE.md
│   └── diagrams/               #   rendered PNG flowcharts
│
└── tensorflow/                 # TensorFlow / Keras implementation
    ├── attention.py, transformer.py, model.py
    ├── dataset.py, train.py, generate.py, chat.py
    ├── paths.py                #   locates ../common
    ├── requirements.txt        #   tensorflow
    ├── Dockerfile, docker-compose.yml
    ├── README.md, ARCHITECTURE.md, NEURAL_NETWORK.md, FLOW.md, DESIGN.md, DOCKER.md
    ├── diagrams/               #   rendered PNGs
    └── checkpoints/            #   trained model (after training)
```

**How sharing works:** both `pytorch/` and `tensorflow/` import the tokenizer and read
the data from `common/`. A tiny `paths.py` in each folder locates `common/` automatically,
so scripts work no matter where you run them from.

---

## Quick start

### PyTorch
```bash
cd minigpt/pytorch
uv run python train.py
uv run python chat.py
```

Or from repo root:
```bash
uv run python minigpt/pytorch/train.py
uv run python minigpt/pytorch/chat.py
```

### TensorFlow
```bash
cd minigpt/tensorflow
uv sync --extra minigpt-tensorflow
uv run python train.py
uv run python chat.py
```

Both produce:
```
You: twinkle
MiniGPT: twinkle little star how i wonder what you are ...
```

---

## Which to read first?

| If you want to... | Go to |
|-------------------|-------|
| **Understand the 10 steps to build an LLM** (start here) | [`common/HOW_TO_BUILD_AN_LLM.md`](common/HOW_TO_BUILD_AN_LLM.md) |
| Learn the model in PyTorch | [`pytorch/README.md`](pytorch/README.md) |
| Learn the model in TensorFlow/Keras | [`tensorflow/README.md`](tensorflow/README.md) |
| Compare the two frameworks | [`common/PYTORCH_VS_TENSORFLOW.md`](common/PYTORCH_VS_TENSORFLOW.md) |
| Understand how it relates to ChatGPT | [`common/COMPARISON.md`](common/COMPARISON.md) |
| **Fine-tune Llama 3 on medical data** | [`finetune/COMPLETE_MANUAL.md`](finetune/COMPLETE_MANUAL.md) |
| See neuron/parameter counts | `pytorch/NEURAL_NETWORK.md` or `tensorflow/NEURAL_NETWORK.md` |
| See flow diagrams | `pytorch/FLOW.md` or `tensorflow/FLOW.md` |

---

## The 10 parts (same in both versions)

1. Dataset & tokenizer → `common/tokenizer.py`, `dataset.py`
2. Token embeddings → `model.py`
3. Positional embeddings → `model.py`
4. Self-attention → `attention.py`
5. Multi-head attention → `attention.py`
6. Transformer block → `transformer.py`
7. Stack into GPT → `model.py`
8. Training → `train.py`
9. Generation → `generate.py`
10. Terminal chat → `chat.py`
