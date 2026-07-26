# MiniGPT (TensorFlow) — Architecture & Design

The TensorFlow/Keras implementation of MiniGPT. Same decoder-only Transformer as the
PyTorch version; this document maps every concept to its Keras building block.

---

## Overview

MiniGPT is a **decoder-only Transformer** trained on *Twinkle Twinkle Little Star*.

| Concept | Where It Lives (Keras) |
|---------|------------------------|
| Tokenization | `tokenizer.py` |
| Token embeddings | `model.py` → `tf.keras.layers.Embedding` |
| Positional embeddings | `model.py` → `tf.keras.layers.Embedding` |
| Scaled dot-product attention | `attention.py` |
| Multi-head attention | `attention.py` → `MultiHeadAttention(Layer)` |
| Causal masking | `attention.py` → `create_causal_mask` |
| Transformer block | `transformer.py` → `TransformerBlock(Layer)` |
| Residual connections | `transformer.py` |
| Layer normalization | `tf.keras.layers.LayerNormalization` |
| Feed-forward network | `transformer.py` → `FeedForward(Layer)` |
| Cross-entropy loss | `train.py` → `SparseCategoricalCrossentropy` |
| Text generation | `model.py` → `generate()` |

---

## Model Specifications

| Hyperparameter | Value |
|----------------|-------|
| Vocabulary Size | 120 (max) / 71 (learned) |
| Embedding Size | 64 |
| Attention Heads | 4 |
| Layers | 2 |
| Sequence Length | 32 |
| FF Hidden Size | 256 |
| **Parameters** | **110,720** |

---

## System Architecture

```
Input: token IDs  (batch, 32)
        │
        ▼
 ┌─────────────────┐   ┌──────────────────┐
 │ Embedding       │ + │ Embedding        │   (token + position)
 │ (vocab → 64)    │   │ (position → 64)  │
 └────────┬────────┘   └────────┬─────────┘
          └──────────┬────────────┘
                     ▼
     ┌──────────────────────────────────┐
     │   TransformerBlock × 2           │
     │   LayerNorm → MHA → +            │
     │   LayerNorm → FeedForward → +    │
     └──────────────┬───────────────────┘
                    ▼
           LayerNormalization
                    ▼
           Dense (64 → vocab)
                    ▼
     Output: logits (batch, 32, vocab)
```

---

## Component Deep Dive (Keras specifics)

### 1. `MiniGPT(tf.keras.Model)`
Subclassed model. Weights are created **lazily** on first call, so we run one dummy
forward pass (`model(tf.zeros((1, 32)))`) to materialize them before saving/counting.

### 2. `MultiHeadAttention(tf.keras.layers.Layer)`
Uses four `Dense(embed_size, use_bias=False)` layers for Q/K/V/O. Heads are created by
`tf.reshape` + `tf.transpose` (splitting the 64-dim vector into 4 × 16).

### 3. Causal mask
Built with `tf.linalg.band_part(ones, -1, 0)` (lower triangle). Applied **additively**:
`scores += (1 - mask) * -1e9` — the Keras-friendly way to send blocked positions to ≈0
after softmax.

### 4. `FeedForward(tf.keras.layers.Layer)`
`Dense(256, activation="gelu")` → `Dense(64)`. GELU is passed as the activation directly.

### 5. Training
`model.compile(optimizer=AdamW, loss=SparseCategoricalCrossentropy(from_logits=True))`
then `model.fit(dataset, epochs=...)`. Keras handles the gradient/step loop internally.

---

## File Structure

```
tensorflow/
├── data/song.txt
├── tokenizer.py          # pure Python (shared with PyTorch)
├── dataset.py            # tf.data pipeline
├── attention.py          # Keras Layer: attention
├── transformer.py        # Keras Layer: block
├── model.py              # tf.keras.Model: MiniGPT
├── train.py              # compile() + fit()
├── generate.py           # load_weights + generate
├── chat.py               # interactive chat
├── ARCHITECTURE.md       # this file
├── NEURAL_NETWORK.md     # neuron/parameter details
├── FLOW.md               # call/train/use flows
├── DESIGN.md             # Keras design decisions
├── DOCKER.md             # containerization
└── checkpoints/          # created after training
      minigpt.weights.h5
      config.json
      tokenizer.json
```

---

## Training Pipeline

1. Load `data/song.txt`, build vocab (≤ 120)
2. Create sliding windows via `tf.data.Dataset`
3. Build `MiniGPT` (~110K params)
4. `compile()` with AdamW + cross-entropy
5. `fit()` for N epochs (Keras runs forward/backward/update)
6. `save_weights()` → `checkpoints/minigpt.weights.h5`

---

## Generation Pipeline

1. Encode prompt → token IDs (with `<BOS>`)
2. `model(idx)` → logits for last position
3. Temperature scaling + top-k filtering
4. `tf.random.categorical` samples the next token
5. Append and repeat
6. Decode IDs → text

See `NEURAL_NETWORK.md` for neuron/parameter details and `FLOW.md` for flow diagrams.
