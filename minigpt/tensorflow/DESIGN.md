# MiniGPT (TensorFlow) — Design Document

Design decisions specific to the TensorFlow/Keras implementation. The architecture
goals are the same as the PyTorch version; this covers the framework-specific choices.

---

## Design Principles

### 1. Idiomatic Keras
We use the **high-level Keras workflow** (`compile()` + `fit()`) rather than a manual
training loop, because that is how most TensorFlow code is written. This keeps `train.py`
short and shows the "TensorFlow way".

> A manual `tf.GradientTape` loop is also possible (and mirrors PyTorch). We chose `fit()`
> for clarity; see `DESIGN.md` exercises below to try the tape version.

### 2. Subclassed Model & Layers
`MiniGPT(tf.keras.Model)` and the custom `Layer` classes mirror the PyTorch `nn.Module`
structure 1:1, so the two codebases read almost the same.

### 3. Same tokenizer, byte-for-byte
`tokenizer.py` is pure Python and shared with the PyTorch version. Tokenization is not a
framework concern.

### 4. `tf.data` input pipeline
We pre-build all sliding windows and use `from_tensor_slices`, which is cleanly
re-iterable each epoch (avoids generator edge cases) and enables `.shuffle()`,
`.batch()`, and `.prefetch()`.

---

## Key TensorFlow/Keras decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Training API | `model.fit()` | Idiomatic, concise |
| Loss | `SparseCategoricalCrossentropy(from_logits=True)` | Targets are integer IDs, model outputs raw logits |
| Optimizer | `tf.keras.optimizers.AdamW` | Matches the PyTorch AdamW |
| Masking | additive (`(1-mask)*-1e9`) | Natural in TF (no boolean masked_fill) |
| Weight init | Keras defaults (Glorot) | Simpler; converges the same |
| Save format | `.weights.h5` + `config.json` | Subclassed models save weights; config rebuilds the graph |
| Activation | `Dense(activation="gelu")` | Built-in, one line |

---

## Why weights are built lazily

Subclassed Keras models don't know their weight shapes until they see input. So we run:

```python
model(tf.zeros((1, SEQ_LEN), dtype=tf.int32))
```

once before `count_params()` or `save_weights()`. This "materializes" all layers.

---

## Differences vs the PyTorch version (intentional)

| Aspect | PyTorch | TensorFlow (here) |
|--------|---------|-------------------|
| Training loop | manual | `fit()` |
| Weight init | explicit `normal_(0.02)` | Keras defaults |
| Dense weight layout | `(out, in)` | `(in, out)` |
| Save | single `.pt` | `.weights.h5` + `config.json` |

These do not change the architecture or the parameter count (still **110,720**).

---

## Suggested Exercises

1. **Manual GradientTape** — rewrite `train.py` with a `tf.GradientTape` loop instead of
   `fit()`. Compare the two styles.
2. **Match PyTorch init** — add a custom initializer (`RandomNormal(stddev=0.02)`) to the
   `Dense`/`Embedding` layers and see if it changes convergence.
3. **Keras built-in attention** — replace the custom `MultiHeadAttention` with
   `tf.keras.layers.MultiHeadAttention` and confirm the output matches.
4. **Callbacks** — add `tf.keras.callbacks.ModelCheckpoint` to save the best model, or
   `TensorBoard` to visualize the loss curve.
5. **`@tf.function`** — wrap the generation loop and measure the speed-up from graph mode.

---

## Limitations (by design)

Same as the PyTorch version: tiny dataset, word-level tokenizer, fixed sequence length,
no KV-cache. All intentional for learning. See the root `DESIGN.md` for the shared
rationale and the comparison to real GPT models.
