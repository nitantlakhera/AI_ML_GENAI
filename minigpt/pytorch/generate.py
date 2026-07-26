"""
Part 9: Text Generation
-----------------------
Load a trained model and generate text from a prompt.

FLOW: prompt text -> token IDs -> model predicts next tokens -> decode back to text.
"""

import argparse                                       # Command-line argument parsing
from pathlib import Path                              # File paths

import torch                                          # PyTorch core

import paths  # noqa: F401  (adds the shared common/ folder to the import path)
from model import MiniGPT                             # The model class
from tokenizer import Tokenizer, BOS_ID               # Tokenizer + the <BOS> token ID

CHECKPOINT_DIR = Path("checkpoints")                  # Default checkpoint folder


def load_model(checkpoint_path: str | Path, device: str) -> tuple[MiniGPT, Tokenizer]:
  # Rebuild the model from a saved checkpoint and load its tokenizer.
  checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)  # Load the .pt file
  config = checkpoint["config"]                       # The saved hyperparameters

  model = MiniGPT(**config).to(device)               # Rebuild the model with the SAME config
  model.load_state_dict(checkpoint["model_state_dict"])  # Load the trained weights into it
  model.eval()                                        # Put model in "evaluation mode" (no dropout, etc.)

  tokenizer_path = Path(checkpoint_path).parent / "tokenizer.json"  # tokenizer.json sits next to it
  tokenizer = Tokenizer.load(tokenizer_path)          # Load the saved vocabulary

  return model, tokenizer                             # Return both, ready to use


def generate_text(
  prompt: str,                                        # The starting text
  max_tokens: int = 20,                               # How many tokens to generate
  temperature: float = 0.8,                           # Randomness (lower = safer)
  top_k: int = 10,                                    # Sample only from the top-k tokens
  checkpoint: str = "checkpoints/minigpt.pt",         # Model checkpoint path
  device: str | None = None,                          # cpu/cuda (auto if None)
) -> str:
  device = device or ("cuda" if torch.cuda.is_available() else "cpu")  # Pick device
  model, tokenizer = load_model(checkpoint, device)   # Load model + tokenizer

  ids = [BOS_ID] + tokenizer.encode(prompt)           # Encode prompt, prefixed with <BOS>
  idx = torch.tensor([ids], dtype=torch.long, device=device)  # Wrap in a batch tensor: (1, len)

  output = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)  # Generate
  generated_ids = output[0].tolist()                  # Take the first (only) sequence -> Python list

  prompt_len = len(ids)                               # Length of the original prompt (to skip it)
  new_ids = generated_ids[prompt_len:]                # Keep only the newly generated tokens
  return tokenizer.decode(new_ids)                    # Decode IDs back into readable text


if __name__ == "__main__":                            # Runs only when executed directly
  parser = argparse.ArgumentParser(description="Generate text with MiniGPT")  # CLI parser
  parser.add_argument("prompt", type=str, help="Starting text prompt")        # Positional: the prompt
  parser.add_argument("--max-tokens", type=int, default=20, help="Tokens to generate")  # --max-tokens
  parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")  # --temperature
  parser.add_argument("--top-k", type=int, default=10, help="Top-k sampling")  # --top-k
  parser.add_argument("--checkpoint", default="checkpoints/minigpt.pt")        # --checkpoint
  parser.add_argument("--device", default=None)                                # --device
  args = parser.parse_args()                          # Parse arguments

  result = generate_text(                             # Generate text using the arguments
    args.prompt,
    max_tokens=args.max_tokens,
    temperature=args.temperature,
    top_k=args.top_k,
    checkpoint=args.checkpoint,
    device=args.device,
  )
  print(f"Prompt:  {args.prompt}")                    # Echo the prompt
  print(f"Output:  {result}")                         # Print the generated text
