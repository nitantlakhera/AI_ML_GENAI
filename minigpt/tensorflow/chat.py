"""
Part 10: Interactive Terminal Chat  (TensorFlow / Keras)
--------------------------------------------------------
Chat with your trained MiniGPT in the terminal.

Example:
  You: twinkle
  MiniGPT: twinkle little star
"""

import argparse                                       # Command-line arguments
import sys                                            # For sys.exit()
from pathlib import Path                              # File path checks

from generate import load_model, generate_text        # Reuse Part 9 helpers

# ASCII banner shown at startup.
BANNER = """
╔══════════════════════════════════════════╗
║       MiniGPT (TensorFlow) — Chat        ║
║   A tiny GPT trained on Twinkle Star     ║
╠══════════════════════════════════════════╣
║  Commands:                               ║
║    quit / exit  — leave chat             ║
║    temp <n>     — set temperature        ║
║    tokens <n>   — set max new tokens     ║
╚══════════════════════════════════════════╝
"""


def chat(checkpoint_dir="checkpoints", temperature=0.7, max_tokens=15):
  ckpt = Path(checkpoint_dir)                          # Checkpoint folder
  if not (ckpt / "minigpt.weights.h5").exists():       # No trained model?
    print("No trained model found. Run training first:")  # Tell the user
    print("  python train.py")
    sys.exit(1)                                        # Exit with error

  print(BANNER)                                        # Show banner
  print(f"Temperature: {temperature} | Max tokens: {max_tokens}\n")  # Settings

  # Load once up front so replies are fast.
  model, tokenizer = load_model(checkpoint_dir)        # Load model + tokenizer

  while True:                                          # Main chat loop
    try:
      user_input = input("You: ").strip()             # Read user text
    except (EOFError, KeyboardInterrupt):             # Handle Ctrl+C / Ctrl+D
      print("\nGoodbye!")
      break

    if not user_input:                                 # Empty line -> ask again
      continue

    if user_input.lower() in ("quit", "exit"):         # Quit command
      print("Goodbye!")
      break

    if user_input.lower().startswith("temp "):         # Change temperature
      try:
        temperature = float(user_input.split()[1])
        print(f"Temperature set to {temperature}")
      except (IndexError, ValueError):
        print("Usage: temp 0.8")
      continue

    if user_input.lower().startswith("tokens "):       # Change max tokens
      try:
        max_tokens = int(user_input.split()[1])
        print(f"Max tokens set to {max_tokens}")
      except (IndexError, ValueError):
        print("Usage: tokens 20")
      continue

    # Generate a reply. We reuse the already-loaded model via generate_text's
    # loader, but to keep it simple we call generate_text (it reloads quickly).
    response = generate_text(
      user_input,
      max_tokens=max_tokens,
      temperature=temperature,
      checkpoint_dir=checkpoint_dir,
    )
    print(f"MiniGPT: {response}\n")                    # Print reply


if __name__ == "__main__":                            # Run only when executed directly
  parser = argparse.ArgumentParser(description="Chat with MiniGPT (TensorFlow)")
  parser.add_argument("--checkpoint-dir", default="checkpoints")
  parser.add_argument("--temperature", type=float, default=0.7)
  parser.add_argument("--max-tokens", type=int, default=15)
  args = parser.parse_args()                          # Parse arguments

  chat(                                                # Start chatting
    checkpoint_dir=args.checkpoint_dir,
    temperature=args.temperature,
    max_tokens=args.max_tokens,
  )
