# MiniGPT: PyTorch vs. TensorFlow/Keras

You now have **two complete implementations** of the exact same model:

| Version | Folder | Framework |
|---------|--------|-----------|
| Original | `../pytorch/` | PyTorch |
| Twin | `../tensorflow/` | TensorFlow 2 + Keras 3 |
| Shared | `../common/` | tokenizer + data (used by both) |

Both produce **110,720 parameters**, train on the same song, and generate the same
kind of output. This document compares them so you can learn both frameworks.

> **Verified:** both versions were trained and generate correctly
> (`twinkle` → "twinkle little star how i wonder what you are ...").

---

## 1. Side-by-side results

| | PyTorch | TensorFlow |
|---|---|---|
| Parameters | 110,720 | 110,720 |
| Start loss | ~2.9 | ~2.9 |
| Final loss (300 epochs) | ~0.04 | ~0.038 |
| `twinkle` → | twinkle little star how i wonder... | twinkle little star how i wonder... |

---

## 2. API translation cheat-sheet

| Concept | PyTorch | TensorFlow / Keras |
|---------|---------|--------------------|
| Base model class | `nn.Module` | `tf.keras.Model` |
| Base layer class | `nn.Module` | `tf.keras.layers.Layer` |
| Forward method | `def forward(self, x)` | `def call(self, x)` |
| Linear layer | `nn.Linear(in, out)` | `tf.keras.layers.Dense(out)` |
| Embedding | `nn.Embedding(v, d)` | `tf.keras.layers.Embedding(v, d)` |
| LayerNorm | `nn.LayerNorm(d)` | `tf.keras.layers.LayerNormalization()` |
| GELU | `nn.GELU()` | `Dense(..., activation="gelu")` |
| Softmax | `F.softmax(x, dim=-1)` | `tf.nn.softmax(x, axis=-1)` |
| Matmul | `torch.matmul(a, b)` | `tf.matmul(a, b)` |
| Transpose | `x.transpose(-2, -1)` | `tf.matmul(..., transpose_b=True)` |
| Reshape | `x.view(...)` | `tf.reshape(x, ...)` |
| Permute dims | `x.transpose(1, 2)` | `tf.transpose(x, perm=[...])` |
| Range | `torch.arange(n)` | `tf.range(n)` |
| Lower triangle | `torch.tril(...)` | `tf.linalg.band_part(..., -1, 0)` |
| Masking | `masked_fill(~mask, -inf)` | `scores += (1-mask) * -1e9` |
| Sampling | `torch.multinomial(p, 1)` | `tf.random.categorical(logits, 1)` |
| Top-k | `torch.topk(x, k)` | `tf.math.top_k(x, k)` |
| Cross-entropy | `F.cross_entropy(logits, y)` | `SparseCategoricalCrossentropy(from_logits=True)` |
| Optimizer | `torch.optim.AdamW` | `tf.keras.optimizers.AdamW` |
| Save | `torch.save(state, path)` | `model.save_weights(path)` |
| Load | `load_state_dict(...)` | `model.load_weights(path)` |

---

## 3. The BIGGEST difference: the training loop

### PyTorch — manual loop (you control every step)
```python
for x, y in loader:
    optimizer.zero_grad()      # clear old gradients
    loss = model.loss(x, y)    # forward + loss
    loss.backward()            # compute gradients
    optimizer.step()           # update weights
```

### TensorFlow/Keras — high-level `fit()` (Keras does it for you)
```python
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
model.fit(dataset, epochs=500)   # <-- zero_grad/backward/step happen internally
```

> Keras also supports a **manual loop** with `tf.GradientTape` if you want the
> PyTorch-style control. This project uses `fit()` because it's the idiomatic Keras way.

---

## 4. Same code, different framework — attention example

### PyTorch (`attention.py`)
```python
d_k = q.size(-1)
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
scores = scores.masked_fill(~mask, float("-inf"))
weights = F.softmax(scores, dim=-1)
return torch.matmul(weights, v)
```

### TensorFlow (`tensorflow/attention.py`)
```python
d_k = tf.cast(tf.shape(q)[-1], tf.float32)
scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(d_k)
scores += (1.0 - mask) * -1e9
weights = tf.nn.softmax(scores, axis=-1)
return tf.matmul(weights, v)
```

**Same math, ~same lines.** Only the function names and the masking style differ.

---

## 5. Masking difference (important detail)

| | PyTorch | TensorFlow |
|---|---|---|
| Mask type | Boolean (`True`=keep) | Float (`1.0`=keep) |
| How applied | `masked_fill(~mask, -inf)` | `scores += (1 - mask) * -1e9` |
| Effect | Blocked → `-inf` → 0 after softmax | Blocked → very negative → ~0 after softmax |

Both achieve the same result: a token cannot attend to future tokens.

---

## 6. Weight initialization note

- **PyTorch** version explicitly initializes weights (`normal_(std=0.02)`) in `_init_weights()`.
- **Keras** uses its own sensible defaults (Glorot/Xavier for `Dense`), so the TF
  version relies on those. This is why the two may not be *bit-identical*, but they
  converge to the same behavior.

---

## 7. File-by-file mapping

| PyTorch file | TensorFlow file | Main change |
|--------------|-----------------|-------------|
| `common/tokenizer.py` | `common/tokenizer.py` | **Shared** — one file used by both (pure Python) |
| `pytorch/dataset.py` | `tensorflow/dataset.py` | `DataLoader` → `tf.data.Dataset` |
| `attention.py` | `tensorflow/attention.py` | `nn.Module` → `keras.layers.Layer` |
| `transformer.py` | `tensorflow/transformer.py` | same structure, Keras layers |
| `model.py` | `tensorflow/model.py` | `nn.Module` → `keras.Model` |
| `train.py` | `tensorflow/train.py` | manual loop → `model.fit()` |
| `generate.py` | `tensorflow/generate.py` | `.pt` load → `load_weights` |
| `chat.py` | `tensorflow/chat.py` | same UX |

---

## 8. Which should you learn?

| | PyTorch | TensorFlow/Keras |
|---|---|---|
| Dominant in LLM research | ✅ | |
| Very concise training (`fit()`) | | ✅ |
| Explicit / "see everything" | ✅ | (use GradientTape) |
| Production/serving tooling | | ✅ (TF Serving, TFLite) |
| Most GPT tutorials use it | ✅ | |

**Recommendation:** Learn both — they express the *same ideas*. Understanding one
makes the other easy. This project lets you compare them directly.

---

## 9. Run them both

```bash
# PyTorch version
cd pytorch
python train.py
python chat.py

# TensorFlow version
cd tensorflow
python train.py
python chat.py
```

Compare the code in each folder side by side — the concepts (embeddings, attention,
masking, residuals, cross-entropy, sampling) are identical.
