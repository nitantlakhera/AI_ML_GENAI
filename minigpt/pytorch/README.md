# MiniGPT — User Manual

A tiny GPT trained from scratch on *Twinkle Twinkle Little Star*. Built for learning how transformers work.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (~1-2 minutes on CPU)
python train.py

# 3. Chat interactively
python chat.py
```

---

## Prerequisites

- **Python 3.10+**
- **PyTorch 2.0+** (CPU is fine; CUDA optional)

---

## Project Structure

```
mini_gpt/
├── data/song.txt       # Training data
├── tokenizer.py        # Text ↔ token IDs
├── dataset.py          # Training data loader
├── attention.py        # Self-attention & multi-head attention
├── transformer.py      # Transformer block
├── model.py            # Full MiniGPT model
├── train.py            # Training script
├── generate.py         # One-shot text generation
├── chat.py             # Interactive chat
├── ARCHITECTURE.md     # Technical architecture guide
├── DESIGN.md           # Design decisions & learning path
└── requirements.txt
```

---

## Step-by-Step Guide

### Step 1: Explore the Data

```bash
cat data/song.txt
```

The model trains on the full lyrics of *Twinkle Twinkle Little Star* (~150 words).

### Step 2: Test the Tokenizer

```bash
python tokenizer.py
```

Expected output:
```
Vocabulary size: 52
Encode: 'twinkle twinkle little star' -> [4, 4, 5, 6]
Decode: 'twinkle twinkle little star'
```

### Step 3: Train the Model

```bash
python train.py
```

**Default settings:**
| Setting | Value |
|---------|-------|
| Epochs | 500 |
| Batch size | 8 |
| Learning rate | 0.003 |
| Sequence length | 32 |

**Custom training:**
```bash
python train.py --epochs 1000 --batch-size 4 --lr 0.001
```

Training output:
```
Using device: cpu
Model parameters: 52,168
Vocabulary size: 52
Training samples: 118
Epoch    1/500 | Loss: 4.2156
Epoch   50/500 | Loss: 1.8234
...
Epoch  500/500 | Loss: 0.0891

Training complete! Checkpoint saved to checkpoints/minigpt.pt
```

Checkpoints are saved to:
- `checkpoints/minigpt.pt` — model weights
- `checkpoints/tokenizer.json` — vocabulary

### Step 4: Generate Text

```bash
python generate.py "twinkle"
```

```
Prompt:  twinkle
Output:  twinkle little star
```

**Options:**
```bash
python generate.py "how i wonder" --max-tokens 10 --temperature 0.5 --top-k 5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--max-tokens` | 20 | Number of tokens to generate |
| `--temperature` | 0.8 | Lower = more focused, higher = more random |
| `--top-k` | 10 | Only sample from top-k most likely tokens |
| `--checkpoint` | checkpoints/minigpt.pt | Model path |

### Step 5: Chat Mode

```bash
python chat.py
```

```
╔══════════════════════════════════════════╗
║           MiniGPT — Chat Mode            ║
║   A tiny GPT trained on Twinkle Star     ║
╚══════════════════════════════════════════╝

You: twinkle
MiniGPT: twinkle little star

You: how i wonder
MiniGPT: what you are

You: quit
Goodbye!
```

**In-chat commands:**
| Command | Example | Effect |
|---------|---------|--------|
| `temp <n>` | `temp 0.5` | Set sampling temperature |
| `tokens <n>` | `tokens 20` | Set max tokens per reply |
| `quit` / `exit` | `quit` | Leave chat |

---

## Model Configuration

All hyperparameters are defined in `train.py` and `model.py`:

| Parameter | Value | File |
|-----------|-------|------|
| Vocabulary Size | 120 | `train.py` → `VOCAB_SIZE` |
| Embedding Size | 64 | `train.py` → `EMBED_SIZE` |
| Attention Heads | 4 | `train.py` → `NUM_HEADS` |
| Layers | 2 | `train.py` → `NUM_LAYERS` |
| Sequence Length | 32 | `train.py` → `SEQ_LEN` |
| FF Hidden Size | 256 | `train.py` → `FF_HIDDEN` |

To change model size, edit these constants and retrain.

---

## Training Tips

### Loss Not Decreasing?
- Increase epochs: `python train.py --epochs 1000`
- Lower learning rate: `python train.py --lr 0.001`

### Output Is Repetitive?
- Increase temperature in chat: `temp 1.0`
- Increase top-k: `python chat.py --temperature 0.9`

### Want Better Quality?
- Train longer (1000+ epochs)
- Add more text to `data/song.txt`
- Increase model size (more layers, larger embeddings)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No trained model found` | Run `python train.py` first |
| `ModuleNotFoundError: torch` | Run `pip install -r requirements.txt` |
| Nonsensical output | Train longer or lower temperature |
| CUDA out of memory | Use CPU: `python train.py --device cpu` |

---

## Learning Resources

| Document | Contents |
|----------|----------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System diagrams, data flow, component details |
| [NEURAL_NETWORK.md](NEURAL_NETWORK.md) | Neuron counts, parameter breakdown, layer diagrams |
| [FLOW.md](FLOW.md) | Call flow, training flow, generation flow (Mermaid diagrams) |
| [../common/COMPARISON.md](../common/COMPARISON.md) | MiniGPT vs ChatGPT: similarities, differences, diagrams |
| [../common/PYTORCH_VS_TENSORFLOW.md](../common/PYTORCH_VS_TENSORFLOW.md) | PyTorch vs TensorFlow/Keras implementations compared |
| [DOCKER.md](DOCKER.md) | Run MiniGPT in Docker (train & chat in a container) |
| [DESIGN.md](DESIGN.md) | Design decisions, exercises, extension ideas |

> This is the **PyTorch** version. The **TensorFlow/Keras** twin lives in [`../tensorflow/`](../tensorflow/),
> and shared code (tokenizer + data) lives in [`../common/`](../common/).

### Recommended Reading Order

1. Read `tokenizer.py` — understand how text becomes numbers
2. Read `attention.py` — the heart of the transformer
3. Read `transformer.py` — how attention fits in a block
4. Read `model.py` — how blocks stack into GPT
5. Run `train.py` — watch loss decrease
6. Run `chat.py` — see generation in action
7. Read `ARCHITECTURE.md` — connect the dots
8. Try exercises in `DESIGN.md`

---

## Example Session

```bash
$ pip install -r requirements.txt
$ python train.py --epochs 500
Using device: cpu
Model parameters: 52,168
...
Training complete!

$ python chat.py
You: twinkle
MiniGPT: twinkle little star

You: like a diamond
MiniGPT: in the sky

You: how i wonder
MiniGPT: what you are

You: quit
Goodbye!
```

---

## License

This project is for educational use. Feel free to modify, extend, and share.
