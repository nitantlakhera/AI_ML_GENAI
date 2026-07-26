"""
Fine-tuning configuration for Llama 3 medical SFT (LoRA / QLoRA).
Override any value via CLI flags in train.py.
"""

from dataclasses import dataclass, field
from pathlib import Path


FINETUNE_ROOT = Path(__file__).resolve().parent
DATA_DIR = FINETUNE_ROOT / "data"
OUTPUT_DIR = FINETUNE_ROOT / "output"


@dataclass
class TrainConfig:
    # --- Model ---
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    use_4bit: bool = True  # QLoRA (needs CUDA + bitsandbytes)
    trust_remote_code: bool = False

    # --- Data ---
    data_path: str = str(DATA_DIR / "medical_sft.jsonl")
    max_seq_length: int = 512
    val_split: float = 0.1  # fraction held out for validation

    # --- LoRA ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # --- Training ---
    output_dir: str = str(OUTPUT_DIR / "llama3-medical-lora")
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    seed: int = 42

    # --- Generation (inference defaults) ---
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    def ensure_dirs(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
