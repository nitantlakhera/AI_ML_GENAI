# MiniGPT — Flow & Call Diagrams

This document shows **how the code runs**: the high-level flow, the training path,
the inference/chat path, and which function calls which.

Diagrams use [Mermaid](https://mermaid.js.org/) and render automatically in Cursor,
GitHub, and most Markdown viewers.

---

## 1. Big picture — two modes

MiniGPT has exactly two ways it is used:

```mermaid
flowchart LR
    subgraph TRAIN["MODE A: TRAIN (once)"]
        S1["data/song.txt"] --> S2["train.py"] --> S3["checkpoints/minigpt.pt<br/>+ tokenizer.json"]
    end
    subgraph USE["MODE B: USE (many times)"]
        U1["your prompt"] --> U2["chat.py / generate.py"] --> U3["generated text"]
    end
    S3 -.loads.-> U2
```

- **Train once** → produces a checkpoint.
- **Use many times** → loads that checkpoint to generate text.

---

## 2. File dependency graph (who imports whom)

```mermaid
flowchart TD
    tok["tokenizer.py<br/>(Part 1)"]
    ds["dataset.py<br/>(Part 1)"]
    att["attention.py<br/>(Parts 4-5)"]
    tf["transformer.py<br/>(Part 6)"]
    mdl["model.py<br/>(Parts 2,3,7)"]
    tr["train.py<br/>(Part 8)"]
    gen["generate.py<br/>(Part 9)"]
    chat["chat.py<br/>(Part 10)"]

    tok --> ds
    att --> tf
    tf --> mdl
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
    A["Start: python train.py"] --> B["create_dataloader()<br/>build tokenizer + sliding windows"]
    B --> C["MiniGPT(...) build model<br/>~110K parameters"]
    C --> D["AdamW optimizer"]
    D --> E["save tokenizer.json"]
    E --> F{"for epoch in 1..N"}
    F --> G{"for each batch (x, y)"}
    G --> H["logits = model(x)  (forward)"]
    H --> I["loss = cross_entropy(logits, y)"]
    I --> J["loss.backward()  (gradients)"]
    J --> K["optimizer.step()  (update weights)"]
    K --> G
    G -- batch done --> L["print avg loss every 50 epochs"]
    L --> F
    F -- all epochs done --> M["torch.save(minigpt.pt)"]
    M --> N["Done"]
```

### What one training step actually does

```mermaid
sequenceDiagram
    participant Trainer
    participant Model as MiniGPT
    participant Optimizer as AdamW
    Trainer->>Optimizer: zero_grad clears old gradients
    Trainer->>Model: compute loss on x and y
    Model->>Model: forward pass yields logits
    Model->>Model: cross entropy yields loss value
    Model-->>Trainer: return loss
    Trainer->>Model: backward computes gradients
    Trainer->>Optimizer: step updates the weights
```

---

## 4. FORWARD PASS flow (`model(x)`)

This is the heart of the network — used in BOTH training and generation.

```mermaid
flowchart TD
    A["idx: token IDs (batch, seq)"] --> B["token_embedding(idx)"]
    A --> C["pos_embedding(0..seq-1)"]
    B --> D(("x = tok + pos"))
    C --> D
    D --> E["Block 1: LN -> Attention -> +  -> LN -> FeedForward -> +"]
    E --> F["Block 2: LN -> Attention -> +  -> LN -> FeedForward -> +"]
    F --> G["Final LayerNorm"]
    G --> H["head: Linear(64 -> vocab)"]
    H --> I["logits (batch, seq, vocab)"]
```

---

## 5. GENERATION / USE flow (`generate.py`, `chat.py`)

```mermaid
flowchart TD
    A["prompt text e.g. 'twinkle'"] --> B["load_model()<br/>rebuild MiniGPT + load weights"]
    B --> C["tokenizer.encode(prompt)<br/>prepend <BOS>"]
    C --> D["idx tensor"]
    D --> E{"repeat max_new_tokens times"}
    E --> F["logits = model(idx)[:, -1, :]<br/>(last position only)"]
    F --> G["apply temperature"]
    G --> H["apply top-k filter"]
    H --> I["softmax -> probabilities"]
    I --> J["sample next token (multinomial)"]
    J --> K["append token to idx"]
    K --> E
    E -- done --> L["tokenizer.decode(new tokens)"]
    L --> M["return generated text"]
```

### Autoregressive loop (token-by-token)

```
Step 1:  [BOS] twinkle              -> predicts "twinkle"
Step 2:  [BOS] twinkle twinkle      -> predicts "little"
Step 3:  [BOS] twinkle twinkle little -> predicts "star"
...each new token is fed back in to predict the next.
```

---

## 6. CHAT call flow (`python chat.py`)

```mermaid
sequenceDiagram
    participant User
    participant Chat as chat.py
    participant Gen as generate.py
    participant Model as MiniGPT

    User->>Chat: python chat.py
    Chat->>Chat: check checkpoint exists
    Chat->>Gen: load_model()
    Gen->>Model: rebuild + load weights
    loop until "quit"
        User->>Chat: types a prompt
        alt command (temp/tokens/quit)
            Chat->>Chat: handle command
        else normal prompt
            Chat->>Gen: generate_text(prompt)
            Gen->>Model: model.generate(...)
            Model-->>Gen: token IDs
            Gen-->>Chat: decoded text
            Chat-->>User: "MiniGPT: ..."
        end
    end
```

---

## 7. Data transformation summary (shapes at each stage)

| Stage | Example | Shape |
|-------|---------|-------|
| Raw text | `"twinkle little star"` | string |
| After `encode()` | `[2, 4, 9, 12]` | list[int] |
| Batched tensor | `[[2, 4, 9, 12]]` | (batch=1, seq=4) |
| After embeddings | vectors | (1, 4, 64) |
| After 2 blocks | vectors | (1, 4, 64) |
| After output head | logits | (1, 4, V) |
| After softmax + sample | next token id | (1, 1) |
| After `decode()` | `"how"` | string |

---

## 8. End-to-end quick reference

```mermaid
flowchart LR
    A["1. Edit data/song.txt<br/>(optional)"] --> B["2. python train.py"]
    B --> C["3. checkpoints/minigpt.pt created"]
    C --> D["4. python chat.py<br/>or python generate.py 'prompt'"]
    D --> E["5. Read the text output"]
```

See **README.md** for the exact commands and **NEURAL_NETWORK.md** for the neuron/parameter details.
