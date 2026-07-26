"""
Part 9: Text Generation  (TensorFlow / Keras)
---------------------------------------------
Load a trained model and generate text from a prompt.
"""

import argparse                                       # Command-line arguments
import json                                           # Read the saved config
from pathlib import Path                              # File paths

import tensorflow as tf                               # TensorFlow / Keras

import paths  # noqa: F401  (adds the shared common/ folder to the import path)
from model import MiniGPT                             # The model class
from tokenizer import Tokenizer, BOS_ID               # Tokenizer + <BOS> ID

CHECKPOINT_DIR = Path("checkpoints")                  # Default checkpoint folder


def load_model(checkpoint_dir="checkpoints"):         # Rebuild + load a trained model
  checkpoint_dir = Path(checkpoint_dir)               # Path object
  config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))  # Read config

  model = MiniGPT(**config)                           # Rebuild with the SAME config
  model(tf.zeros((1, config["seq_len"]), dtype=tf.int32))  # Dummy pass builds the weights
  model.load_weights(str(checkpoint_dir / "minigpt.weights.h5"))  # Load trained weights

  tokenizer = Tokenizer.load(checkpoint_dir / "tokenizer.json")   # Load vocabulary
  return model, tokenizer                             # Return both


def generate_text(
  prompt: str,                                        # Starting text
  max_tokens: int = 20,                               # Tokens to generate
  temperature: float = 0.8,                           # Randomness
  top_k: int = 10,                                    # Top-k sampling
  checkpoint_dir: str = "checkpoints",                # Where the model lives
) -> str:
  model, tokenizer = load_model(checkpoint_dir)       # Load model + tokenizer

  ids = [BOS_ID] + tokenizer.encode(prompt)           # Encode prompt with <BOS>
  idx = tf.constant([ids], dtype=tf.int32)            # Batch tensor: (1, len)

  output = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
  generated_ids = output.numpy()[0].tolist()          # Tensor -> Python list

  new_ids = generated_ids[len(ids):]                  # Keep only newly generated tokens
  return tokenizer.decode(new_ids)                    # Decode back to text


if __name__ == "__main__":                            # Run only when executed directly
  parser = argparse.ArgumentParser(description="Generate text with MiniGPT (TensorFlow)")
  parser.add_argument("prompt", type=str, help="Starting text prompt")
  parser.add_argument("--max-tokens", type=int, default=20, help="Tokens to generate")
  parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
  parser.add_argument("--top-k", type=int, default=10, help="Top-k sampling")
  parser.add_argument("--checkpoint-dir", default="checkpoints")
  args = parser.parse_args()                          # Parse arguments

  result = generate_text(                             # Generate
    args.prompt,
    max_tokens=args.max_tokens,
    temperature=args.temperature,
    top_k=args.top_k,
    checkpoint_dir=args.checkpoint_dir,
  )
  print(f"Prompt:  {args.prompt}")                    # Echo prompt
  print(f"Output:  {result}")                         # Print generated text
