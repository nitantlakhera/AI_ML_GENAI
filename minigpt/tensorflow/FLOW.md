# MiniGPT (TensorFlow) — Flow & Call Diagrams

How the TensorFlow/Keras code runs: training with `model.fit()`, the `tf.data` pipeline,
generation, and chat. Mermaid diagrams render in Cursor/GitHub preview.

---

## 1. Big picture — two modes

```mermaid
flowchart LR
    subgraph TRAIN["MODE A: TRAIN (once)"]
        S1["data/song.txt"] --> S2["train.py<br/>model.fit()"] --> S3["checkpoints/<br/>minigpt.weights.h5 + config.json + tokenizer.json"]
    end
    subgraph USE["MODE B: USE (many times)"]
        U1["your prompt"] --> U2["chat.py / generate.py"] --> U3["generated text"]
    end
    S3 -.load_weights.-> U2
```

---

## 2. File dependency graph

```mermaid
flowchart TD
    tok["tokenizer.py"]
    ds["dataset.py<br/>(tf.data)"]
    att["attention.py<br/>(keras Layer)"]
    tf_["transformer.py<br/>(keras Layer)"]
    mdl["model.py<br/>(keras Model)"]
    tr["train.py"]
    gen["generate.py"]
    chat["chat.py"]

    tok --> ds
    att --> tf_
    tf_ --> mdl
    ds --> tr
    mdl --> tr
    mdl --> gen
    tok --> gen
    gen --> chat
```

---

## 3. TRAINING flow (`python train.py`)

```mermaid
flowchart TD
    A["Start: python train.py"] --> B["create_dataset()<br/>tokenizer + tf.data windows"]
    B --> C["MiniGPT(...) build model"]
    C --> D["dummy forward: model(zeros)<br/>materialize weights"]
    D --> E["model.compile(AdamW, CrossEntropy)"]
    E --> F["model.fit(ds, epochs=N)"]
    F --> G["Keras runs forward + backward + update"]
    G --> H["save_weights(minigpt.weights.h5)<br/>+ config.json + tokenizer.json"]
    H --> I["Done"]
```

### What `model.fit()` does internally per step

```mermaid
sequenceDiagram
    participant Keras as model.fit
    participant Tape as GradientTape
    participant Model as MiniGPT
    participant Optim as AdamW
    Keras->>Tape: open gradient tape
    Tape->>Model: forward pass yields logits
    Model->>Model: cross entropy yields loss value
    Keras->>Tape: tape.gradient of loss
    Tape-->>Keras: gradients
    Keras->>Optim: apply_gradients updates weights
```

> In PyTorch you write this loop by hand (`zero_grad` / `backward` / `step`).
> Keras `fit()` does it for you using a `GradientTape` under the hood.

---

## 4. FORWARD PASS flow (`model(idx)`)

```mermaid
flowchart TD
    A["idx: token IDs (batch, seq)"] --> B["Embedding(idx) token"]
    A --> C["Embedding(range) position"]
    B --> D(("x = tok + pos"))
    C --> D
    D --> E["TransformerBlock 1"]
    E --> F["TransformerBlock 2"]
    F --> G["LayerNormalization"]
    G --> H["Dense (64 -> vocab)"]
    H --> I["logits (batch, seq, vocab)"]
```

---

## 5. GENERATION flow

```mermaid
flowchart TD
    A["prompt text"] --> B["load_model()<br/>rebuild + load_weights"]
    B --> C["tokenizer.encode + BOS"]
    C --> D{"repeat max_new_tokens"}
    D --> E["model(idx) last-position logits"]
    E --> F["temperature scaling"]
    F --> G["top-k filter"]
    G --> H["tf.random.categorical sample"]
    H --> I["append token"]
    I --> D
    D -- done --> J["tokenizer.decode"]
    J --> K["generated text"]
```

---

## 6. CHAT call flow

```mermaid
sequenceDiagram
    participant User
    participant Chat as chat.py
    participant Gen as generate.py
    participant Model as MiniGPT
    User->>Chat: python chat.py
    Chat->>Gen: load_model()
    Gen->>Model: rebuild + load_weights
    loop until quit
        User->>Chat: types a prompt
        Chat->>Gen: generate_text(prompt)
        Gen->>Model: model.generate(...)
        Model-->>Gen: token IDs
        Gen-->>Chat: decoded text
        Chat-->>User: MiniGPT reply
    end
```

---

## 7. Data shapes at each stage

| Stage | Example | Shape |
|-------|---------|-------|
| Raw text | `"twinkle little star"` | string |
| After `encode()` | `[2, 4, 9, 12]` | list[int] |
| Batched tensor | `[[2, 4, 9, 12]]` | (1, 4) int32 |
| After embeddings | vectors | (1, 4, 64) |
| After 2 blocks | vectors | (1, 4, 64) |
| After Dense head | logits | (1, 4, V) |
| After categorical sample | next token id | (1, 1) |
| After `decode()` | `"how"` | string |
