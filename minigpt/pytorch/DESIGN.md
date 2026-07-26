# MiniGPT — Design Document

## Purpose

MiniGPT exists as a **learning vehicle**. Every design choice prioritizes clarity and educational value over performance. The goal is to understand transformers by reading, running, and modifying real code — not by using a black-box library.

---

## Design Principles

### 1. Build Everything From Scratch

No `transformers` library, no pre-trained weights. Every component — attention, layer norm, embeddings — is implemented in plain PyTorch so you can trace exactly what happens at each step.

### 2. Small Enough to Train on CPU

| Choice | Rationale |
|--------|-----------|
| 120 vocab size | Covers the song lyrics with room for special tokens |
| 64 embedding dim | Large enough to learn patterns, small enough for fast training |
| 2 layers | Minimum to demonstrate depth without overfitting complexity |
| 4 attention heads | Shows multi-head concept without excessive parameters |
| 32 sequence length | Fits typical lyric lines; keeps memory low |

Training completes in **under 2 minutes on CPU** with 500 epochs.

### 3. Word-Level Tokenization

Character-level tokenization would produce very long sequences and slow training. Word-level is simpler to understand and matches how we think about lyrics:

```
"twinkle twinkle little star" → [twinkle, twinkle, little, star]
```

Trade-off: the model can only generate known vocabulary words (no invented spellings).

### 4. Learned Positional Embeddings

Two common approaches exist:

| Approach | Pros | Cons |
|----------|------|------|
| Sinusoidal (fixed) | No extra parameters; generalizes to longer sequences | Harder to understand |
| **Learned (chosen)** | Simple `nn.Embedding`; easy to inspect | Fixed max sequence length |

For a learning project with fixed `seq_len=32`, learned embeddings are the clearer choice.

### 5. Pre-LayerNorm (GPT-style)

Modern GPT models apply layer normalization **before** each sublayer:

```
x + Attention(LayerNorm(x))    ← Pre-LN (used here)
vs
LayerNorm(x + Attention(x))    ← Post-LN (original Transformer)
```

Pre-LN trains more stably, especially with few layers.

### 6. GELU Activation

The feed-forward network uses GELU instead of ReLU. GELU is smoother and is the standard in GPT-2/3/4:

```
GELU(x) ≈ x · Φ(x)    where Φ is the Gaussian CDF
```

### 7. Causal (Left-to-Right) Attention Only

MiniGPT is a **decoder-only** model. There is no encoder, no cross-attention. This matches the GPT architecture used in ChatGPT.

---

## Learning Path

Follow the parts in order. Each file maps to one or more parts:

```
Part 1  → tokenizer.py, dataset.py
Part 2  → model.py (token_embedding)
Part 3  → model.py (pos_embedding)
Part 4  → attention.py (scaled_dot_product_attention)
Part 5  → attention.py (MultiHeadAttention)
Part 6  → transformer.py
Part 7  → model.py (blocks stack)
Part 8  → train.py
Part 9  → generate.py
Part 10 → chat.py
```

### Suggested Exercises

After running the default pipeline, try these modifications to deepen understanding:

1. **Change attention heads** — Set `NUM_HEADS = 1` in `train.py`. How does loss change?
2. **Remove causal mask** — Comment out masking in `attention.py`. Does the model still learn?
3. **Increase layers** — Try `NUM_LAYERS = 4`. Does it overfit faster?
4. **Temperature experiment** — Generate with `temperature=0.1` vs `temperature=1.5` in `chat.py`
5. **Add your own text** — Replace `data/song.txt` with different lyrics and retrain
6. **Print attention weights** — Modify `scaled_dot_product_attention` to return weights and visualize which tokens attend to which

---

## Limitations (By Design)

| Limitation | Why It's OK |
|------------|-------------|
| Tiny dataset (one song) | Focus is on architecture, not data scale |
| Word-level only | Simpler than BPE/subword tokenization |
| No GPU requirement | Accessible to all learners |
| Fixed sequence length | Avoids complexity of relative position encodings |
| No KV-cache during generation | Simpler code; slower generation is fine at this scale |

These are intentional. A production GPT would address all of them — but understanding the core mechanics comes first.

---

## Comparison to Real GPT Models

| Feature | MiniGPT | GPT-2 Small | GPT-3 |
|---------|---------|-------------|-------|
| Parameters | ~50K | 124M | 175B |
| Layers | 2 | 12 | 96 |
| Embedding dim | 64 | 768 | 12,288 |
| Heads | 4 | 12 | 96 |
| Vocab | 120 | 50,257 | 50,257 |
| Context length | 32 | 1,024 | 2,048 |
| Training data | 1 song | WebText | Internet |

MiniGPT implements the **same mathematical operations** at a scale where you can inspect every tensor.

---

## Extension Ideas

Once comfortable with the basics:

1. **Byte-pair encoding (BPE)** — Subword tokenization like real GPTs use
2. **KV-cache** — Speed up generation by caching key/value tensors
3. **Learning rate schedule** — Warmup + cosine decay
4. **Weight tying** — Share weights between embedding and output head
5. **Larger corpus** — Train on Shakespeare or a book
6. **Attention visualization** — Plot attention heatmaps per head
7. **Web UI** — Replace terminal chat with a Gradio/Streamlit interface
