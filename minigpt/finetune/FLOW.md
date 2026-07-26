# Fine-Tuning Flow Diagrams

> **📖 All diagrams are also in [COMPLETE_MANUAL.md](COMPLETE_MANUAL.md)** — Part IV (sections 22–35).

How data and code flow through the Llama 3 medical fine-tuning pipeline.

Diagrams use [Mermaid](https://mermaid.js.org/) and render in GitHub, Cursor, and most Markdown viewers.
Export PNGs with `python render_diagrams.py`.

---

## 1. Big picture — train once, use many times

```mermaid
flowchart LR
    subgraph TRAIN["MODE A: FINE-TUNE (once)"]
        D1["data/medical_sft.jsonl"] --> T1["train.py<br/>QLoRA SFT"]
        T1 --> O1["output/llama3-medical-lora/<br/>adapter weights"]
    end
    subgraph USE["MODE B: INFERENCE (many times)"]
        U1["your medical question"] --> I1["inference.py"] --> U2["generated answer"]
    end
    O1 -.loads.-> I1
```

---

## 2. File dependency graph

```mermaid
flowchart TD
    cfg["config.py<br/>(hyperparameters)"]
    data["data/medical_sft.jsonl"]
    ds["dataset.py<br/>(JSONL → chat template)"]
    tr["train.py<br/>(QLoRA SFT)"]
    inf["inference.py<br/>(chat)"]
    merge["merge_lora.py<br/>(optional)"]
    hf["Hugging Face Hub<br/>Llama 3 8B Instruct"]
    out["output/llama3-medical-lora/"]

    data --> ds
    cfg --> tr
    ds --> tr
    hf --> tr
    tr --> out
    out --> inf
    hf --> inf
    out --> merge
    hf --> merge
```

---

## 3. Training flow (`python train.py`)

```mermaid
flowchart TD
    A["Start: python train.py"] --> B["Load TrainConfig<br/>(config.py)"]
    B --> C["load_jsonl()<br/>read medical_sft.jsonl"]
    C --> D["train/val split<br/>(90% / 10%)"]
    D --> E["Download Llama 3 8B<br/>from Hugging Face"]
    E --> F{"use_4bit?"}
    F -->|Yes| G["Load in 4-bit<br/>(QLoRA + bitsandbytes)"]
    F -->|No| H["Load in bf16<br/>(full LoRA)"]
    G --> I["Attach LoRA adapters<br/>(q_proj, k_proj, v_proj, o_proj)"]
    H --> I
    I --> J["apply_chat_template()<br/>format each Q&A row"]
    J --> K["SFTTrainer.train()"]
    K --> L["Save adapter + tokenizer<br/>to output/"]
    L --> M["Write run_info.json"]
    M --> N["Done"]
```

---

## 4. What one training step does

```mermaid
sequenceDiagram
    participant Batch as Batch (Q&A text)
    participant Model as Llama 3 + LoRA
    participant Loss as Cross-Entropy Loss
    participant Opt as Optimizer

    Batch->>Model: Token IDs (input window)
    Model->>Model: Embeddings → Transformer blocks → logits
    Note over Model: Only LoRA weights receive gradients
    Model->>Loss: Predicted vs actual next token
    Loss->>Opt: Backpropagation
    Opt->>Model: Update LoRA weights only
```

---

## 5. Inference flow (`python inference.py`)

```mermaid
flowchart TD
    A["Start: python inference.py"] --> B["Load base Llama 3 8B<br/>(4-bit quantized)"]
    B --> C["Load LoRA adapter<br/>from output/"]
    C --> D{"--prompt given?"}
    D -->|Yes| E["Single generation"]
    D -->|No| F["Interactive chat loop"]
    E --> G["apply_chat_template(user message)"]
    F --> G
    G --> H["model.generate()<br/>temperature + top-p sampling"]
    H --> I["Decode new tokens → answer"]
    I --> J["Print response"]
    F --> F
```

---

## 6. Data pipeline — JSONL to training text

```mermaid
flowchart LR
    A["JSONL row"] --> B["instruction + input + output"]
    B --> C["row_to_messages()"]
    C --> D["user / assistant messages"]
    D --> E["tokenizer.apply_chat_template()"]
    E --> F["Llama 3 formatted text"]
    F --> G["SFTTrainer<br/>next-token prediction"]
```

Example transformation:

```
INPUT (JSONL):
  instruction: "What is hypertension?"
  output:      "Hypertension is high blood pressure..."

OUTPUT (chat template text):
  <|begin_of_text|><|start_header_id|>user<|end_header_id|>
  What is hypertension?<|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>
  Hypertension is high blood pressure...<|eot_id|>
```

---

## 7. LoRA vs full fine-tuning

```mermaid
flowchart TD
    subgraph FULL["Full Fine-Tuning"]
        F1["All 8B parameters trainable"] --> F2["Best quality potential"]
        F2 --> F3["Needs 80+ GB VRAM"]
    end
    subgraph LORA["LoRA (this project)"]
        L1["Freeze base model"] --> L2["Train small adapter matrices<br/>(~0.5–2% of params)"]
        L2 --> L3["Needs 8–16 GB VRAM"]
    end
    subgraph QLORA["QLoRA (default)"]
        Q1["Base model in 4-bit"] --> Q2["+ LoRA adapters in fp16"]
        Q2 --> Q3["Needs ~8 GB VRAM"]
    end
```

---

## 8. End-to-end lifecycle

```mermaid
flowchart TD
    S1["1. Prepare medical JSONL"] --> S2["2. huggingface-cli login"]
    S2 --> S3["3. python train.py"]
    S3 --> S4["4. Evaluate answers<br/>(manual or benchmark)"]
    S4 --> S5{"Good enough?"}
    S5 -->|No| S6["Add data / tune hyperparams"]
    S6 --> S3
    S5 -->|Yes| S7["5. python inference.py"]
    S7 --> S8["6. Optional: merge_lora.py<br/>for deployment"]
```

---

## 9. Comparison with MiniGPT training

```mermaid
flowchart TD
    subgraph MINI["MiniGPT (pytorch/train.py)"]
        M1["Random init weights"] --> M2["Train ALL 110K params"]
        M2 --> M3["Tiny song text"]
        M3 --> M4["checkpoints/minigpt.pt"]
    end
    subgraph LLAMA["Llama 3 Fine-tune (finetune/train.py)"]
        L1["Pretrained 8B weights"] --> L2["Train LoRA adapters only"]
        L2 --> L3["Medical Q&A JSONL"]
        L3 --> L4["output/llama3-medical-lora/"]
    end
```
