# MiniGPT — TensorFlow / Keras Version

The **exact same** MiniGPT (Twinkle Twinkle Little Star), reimplemented in
**TensorFlow 2 + Keras 3**. Same architecture, same ~110K parameters, same output —
just a different framework so you can learn both.

> This is the TensorFlow twin of the PyTorch project in [`../pytorch/`](../pytorch/).
> Shared code (tokenizer + data) lives in [`../common/`](../common/).
> See [`../common/PYTORCH_VS_TENSORFLOW.md`](../common/PYTORCH_VS_TENSORFLOW.md) for a line-by-line comparison.

## Documentation

| Document | Contents |
|----------|----------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design, Keras component mapping |
| [NEURAL_NETWORK.md](NEURAL_NETWORK.md) | Neuron counts + parameter breakdown (verified) |
| [FLOW.md](FLOW.md) | Train / forward / generate / chat flow diagrams |
| [DESIGN.md](DESIGN.md) | Keras-specific design decisions + exercises |
| [DOCKER.md](DOCKER.md) | Run the TF version in Docker |
| [diagrams/](diagrams/) | Rendered PNG diagrams (9 images) |
| [../common/PYTORCH_VS_TENSORFLOW.md](../common/PYTORCH_VS_TENSORFLOW.md) | PyTorch vs TensorFlow comparison |

---

## Model specs (identical to the PyTorch version)

| Hyperparameter | Value |
|----------------|-------|
| Vocabulary size | 120 (max) / 71 (learned) |
| Embedding size | 64 |
| Attention heads | 4 |
| Layers | 2 |
| Sequence length | 32 |
| Feed-forward hidden | 256 |
| **Total parameters** | **110,720** |

---

## Quick Start

```bash
# 1. Install TensorFlow
pip install -r requirements.txt

# 2. Train (~1 min on CPU)
python train.py

# 3. Chat
python chat.py
```

---

## Files (mirror the PyTorch layout)

| File | Part | What it does |
|------|------|--------------|
| `data/song.txt` | 1 | Training lyrics |
| `tokenizer.py` | 1 | Word ↔ ID (pure Python — identical to PyTorch) |
| `dataset.py` | 1 | `tf.data` sliding-window pipeline |
| `attention.py` | 4-5 | Self-attention + multi-head (Keras `Layer`) |
| `transformer.py` | 6 | Transformer block (Keras `Layer`) |
| `model.py` | 2,3,7 | Embeddings + full `tf.keras.Model` |
| `train.py` | 8 | Training via `model.compile()` + `model.fit()` |
| `generate.py` | 9 | Text generation |
| `chat.py` | 10 | Interactive chat |

---

## Commands

### Train
```bash
python train.py --epochs 500 --batch-size 8 --lr 0.003
```
Saves:
- `checkpoints/minigpt.weights.h5` — trained weights (Keras format)
- `checkpoints/config.json` — model config
- `checkpoints/tokenizer.json` — vocabulary

### Generate
```bash
python generate.py "twinkle" --max-tokens 20 --temperature 0.8 --top-k 10
```

### Chat
```bash
python chat.py
```
```
You: twinkle
MiniGPT: twinkle little star
You: quit
```

---

## Key Keras concepts used

| Concept | Where |
|---------|-------|
| `tf.keras.Model` subclassing | `model.py` → `MiniGPT` |
| `tf.keras.layers.Layer` subclassing | `attention.py`, `transformer.py` |
| `tf.keras.layers.Dense` (= `nn.Linear`) | Q/K/V/O + feed-forward |
| `tf.keras.layers.Embedding` | token + positional embeddings |
| `tf.keras.layers.LayerNormalization` | inside each block |
| `model.compile()` + `model.fit()` | training (replaces manual loop) |
| `tf.data.Dataset` | input pipeline |
| `tf.random.categorical` | sampling during generation |

---

## Notes

- **CPU is used** on native Windows (TensorFlow GPU needs WSL2). That's fine —
  training takes ~1 minute.
- The informational logs about `oneDNN` and GPU support are harmless.
- To silence TensorFlow's startup logs, set `TF_CPP_MIN_LOG_LEVEL=3` before running.
