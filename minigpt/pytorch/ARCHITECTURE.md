# MiniGPT — Architecture & Design

A from-scratch GPT-style language model for learning how transformers work.

---

## Table of Contents

1. [Overview](#overview)
2. [Model Specifications](#model-specifications)
3. [System Architecture](#system-architecture)
4. [Data Flow](#data-flow)
5. [Component Deep Dive](#component-deep-dive)
6. [File Structure](#file-structure)
7. [Training Pipeline](#training-pipeline)
8. [Generation Pipeline](#generation-pipeline)

---

## Overview

MiniGPT is a **decoder-only transformer** (the same family as GPT-2/3/4) trained on the lyrics of *Twinkle Twinkle Little Star*. Despite its tiny size, it implements every core building block of modern language models:

| Concept | Where It Lives |
|---------|----------------|
| Tokenization | `tokenizer.py` |
| Token embeddings | `model.py` → `token_embedding` |
| Positional embeddings | `model.py` → `pos_embedding` |
| Scaled dot-product attention | `attention.py` |
| Multi-head attention | `attention.py` → `MultiHeadAttention` |
| Causal masking | `attention.py` → `create_causal_mask` |
| Transformer block | `transformer.py` |
| Residual connections | `transformer.py` |
| Layer normalization | `transformer.py` |
| Feed-forward network | `transformer.py` → `FeedForward` |
| Cross-entropy loss | `model.py` → `loss()` |
| Text generation | `model.py` → `generate()` |

---

## Model Specifications

| Hyperparameter | Value | Meaning |
|----------------|-------|---------|
| Vocabulary Size | 120 | Max unique tokens (words + special tokens) |
| Embedding Size | 64 | Dimension of each token vector |
| Attention Heads | 4 | Parallel attention mechanisms |
| Layers | 2 | Stacked transformer blocks |
| Sequence Length | 32 | Max tokens per training window |
| FF Hidden Size | 256 | Feed-forward inner dimension |
| Head Dimension | 16 | `embed_size / num_heads` = 64 / 4 |

**Approximate parameter count:** ~50,000 trainable parameters.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         MiniGPT Model                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: token IDs  [batch, seq_len]                             │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐    ┌──────────────────┐                  │
│  │ Token Embedding │ +  │ Positional Embed │   (Part 2 & 3)     │
│  │  (vocab → 64)   │    │  (position → 64) │                    │
│  └────────┬────────┘    └────────┬─────────┘                  │
│           └──────────┬─────────────┘                            │
│                      ▼                                          │
│  ┌──────────────────────────────────────────┐                     │
│  │         Transformer Block × 2           │   (Part 6 & 7)    │
│  │  ┌──────────────────────────────────┐  │                     │
│  │  │ LayerNorm → Multi-Head Attn → +  │  │                     │
│  │  │ LayerNorm → Feed-Forward    → +  │  │                     │
│  │  └──────────────────────────────────┘  │                     │
│  └──────────────────┬───────────────────────┘                   │
│                     ▼                                           │
│              LayerNorm (final)                                  │
│                     ▼                                           │
│              Linear Head (64 → vocab)                           │
│                     ▼                                           │
│  Output: logits  [batch, seq_len, vocab_size]                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Training Flow

```
song.txt
   │
   ▼
Tokenizer (word → ID)
   │
   ▼
Sliding windows of length 32
   │
   ├── Input:  [t0, t1, ..., t31]
   └── Target: [t1, t2, ..., t32]    ← shifted by 1 (next-token prediction)
   │
   ▼
MiniGPT.forward() → logits
   │
   ▼
CrossEntropyLoss(logits, targets)
   │
   ▼
Backpropagation → update weights
```

### Generation Flow

```
User prompt: "twinkle"
   │
   ▼
Tokenizer.encode() → [BOS, twinkle]
   │
   ▼
Model.forward() → logits for last position
   │
   ▼
Softmax + sample (temperature, top-k)
   │
   ▼
Append token → repeat until max_tokens
   │
   ▼
Tokenizer.decode() → "twinkle little star"
```

---

## Component Deep Dive

### 1. Tokenizer (`tokenizer.py`)

- **Type:** Word-level
- **Special tokens:** `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`
- **Normalization:** Lowercase, strip punctuation
- **Vocabulary:** Built from song text, capped at 120 words

### 2. Self-Attention (`attention.py`)

For each token, attention computes a weighted sum of all previous tokens:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

**Why scaling (√d_k)?** Prevents dot products from growing too large, which would push softmax into regions with near-zero gradients.

### 3. Causal Mask

```
Position:  0   1   2   3
Token 0:  [✓   ✗   ✗   ✗]
Token 1:  [✓   ✓   ✗   ✗]
Token 2:  [✓   ✓   ✓   ✗]
Token 3:  [✓   ✓   ✓   ✓]
```

Token at position `i` can only attend to positions `≤ i`. This is essential for autoregressive generation — the model must not "peek" at future tokens during training.

### 4. Multi-Head Attention

Instead of one attention operation, we run **4 parallel heads** (each with dimension 16). Each head can learn different patterns:
- Head 1 might focus on repeated words ("twinkle twinkle")
- Head 2 might track rhyme ("star" → "are")
- Head 3 might capture line structure

### 5. Transformer Block

```
x ──────────────────────────────────────► (+) ──►
│                                          ▲
└─► LayerNorm ─► Multi-Head Attn ─────────┘
                    │
                    ▼
x ──────────────────────────────────────► (+) ──► output
│                                          ▲
└─► LayerNorm ─► Feed-Forward ────────────┘
```

**Residual connections** (`x + sublayer(x)`) allow gradients to flow directly through the network, making deep models trainable.

**Layer normalization** stabilizes activations by normalizing across the embedding dimension.

### 6. Feed-Forward Network

```
64 → 256 (GELU) → 64
```

A simple two-layer MLP applied independently to each token position. It adds non-linearity and increases model capacity.

---

## File Structure

```
mini_gpt/
│
├── data/
│     song.txt          # Training corpus (Twinkle Twinkle Little Star)
│
├── tokenizer.py        # Part 1: Vocabulary & encode/decode
├── dataset.py          # Part 1: Sliding-window dataset
├── attention.py        # Parts 4-5: Self-attention & multi-head
├── transformer.py      # Part 6: Transformer block
├── model.py            # Parts 2-3, 7: Embeddings + full GPT
├── train.py            # Part 8: Training loop
├── generate.py         # Part 9: Text generation
├── chat.py             # Part 10: Interactive chat
│
├── checkpoints/        # Created after training
│     minigpt.pt        # Model weights
│     tokenizer.json    # Saved vocabulary
│
├── ARCHITECTURE.md     # This file
├── DESIGN.md           # Design decisions & learning guide
├── README.md           # User manual
└── requirements.txt
```

---

## Training Pipeline

1. Load `data/song.txt`
2. Build vocabulary (≤ 120 words)
3. Create sliding-window samples (length 32)
4. Initialize MiniGPT (~50K parameters)
5. For each epoch:
   - Forward pass → compute cross-entropy loss
   - Backward pass → update weights with AdamW
6. Save checkpoint to `checkpoints/minigpt.pt`

**Loss function:** Cross-entropy — measures how well the model predicts the next token.

**Optimizer:** AdamW — adaptive learning rate with weight decay.

---

## Generation Pipeline

1. Encode user prompt to token IDs
2. Feed tokens through model → get logits for last position
3. Apply temperature scaling (lower = more deterministic)
4. Apply top-k filtering (keep only top k probable tokens)
5. Sample next token from probability distribution
6. Append token and repeat
7. Decode token IDs back to text

**Greedy decoding** (temperature → 0) always picks the highest-probability token.
**Sampling** (temperature > 0) introduces randomness for more varied output.

---

## What You'll Learn

By studying and running this project:

1. **Embeddings** — How words become dense vectors that capture meaning
2. **Self-attention** — How tokens communicate with each other
3. **Causal masking** — Why GPT can't see the future
4. **Multi-head attention** — Parallel attention for richer representations
5. **Residual connections** — How deep networks avoid vanishing gradients
6. **Layer normalization** — Training stability
7. **Feed-forward networks** — Per-token computation after attention
8. **Cross-entropy loss** — The standard objective for language modeling
9. **Backpropagation** — How gradients update every weight
10. **Autoregressive generation** — Token-by-token text creation
