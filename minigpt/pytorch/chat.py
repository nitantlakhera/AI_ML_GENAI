"""
Part 10: Interactive Terminal Chat
----------------------------------
Chat with your trained MiniGPT in the terminal.

Example:
  You: twinkle
  MiniGPT: twinkle little star

  You: how i wonder
  MiniGPT: what you are
"""

import argparse                                       # Command-line argument parsing
import sys                                            # For sys.exit() on error
from pathlib import Path                              # File path checks

import torch                                          # PyTorch core

from generate import generate_text, load_model        # Reuse Part 9 helpers

# ASCII banner shown when the chat starts.
BANNER = """
╔══════════════════════════════════════════╗
║           MiniGPT — Chat Mode            ║
║   A tiny GPT trained on Twinkle Star     ║
╠══════════════════════════════════════════╣
║  Commands:                               ║
║    quit / exit  — leave chat             ║
║    temp <n>     — set temperature        ║
║    tokens <n>   — set max new tokens     ║
╚══════════════════════════════════════════╝
"""


def chat(
  checkpoint: str = "checkpoints/minigpt.pt",         # Path to the trained model
  device: str | None = None,                          # cpu/cuda (auto if None)
  temperature: float = 0.7,                           # Default randomness
  max_tokens: int = 15,                               # Default reply length
) -> None:
  device = device or ("cuda" if torch.cuda.is_available() else "cpu")  # Pick device

  if not Path(checkpoint).exists():                   # If no trained model exists yet...
    print("No trained model found. Run training first:")  # ...tell the user
    print("  python train.py")                        # ...how to fix it
    sys.exit(1)                                        # ...and exit with an error code

  print(BANNER)                                        # Show the welcome banner
  print(f"Device: {device} | Temperature: {temperature} | Max tokens: {max_tokens}\n")  # Settings line

  load_model(checkpoint, device)                       # Warm-up load (checks the model is valid)

  while True:                                          # Main chat loop (runs until user quits)
    try:
      user_input = input("You: ").strip()             # Read one line of user input
    except (EOFError, KeyboardInterrupt):             # Handle Ctrl+C / Ctrl+D gracefully
      print("\nGoodbye!")                             # Say goodbye
      break                                            # Exit the loop

    if not user_input:                                 # If the user pressed Enter with no text...
      continue                                         # ...ask again

    if user_input.lower() in ("quit", "exit"):         # If the user typed quit/exit...
      print("Goodbye!")                                # ...say goodbye
      break                                            # ...and leave the loop

    if user_input.lower().startswith("temp "):         # Command: change temperature
      try:
        temperature = float(user_input.split()[1])     # Parse the number after "temp "
        print(f"Temperature set to {temperature}")     # Confirm the new value
      except (IndexError, ValueError):                 # If parsing failed...
        print("Usage: temp 0.8")                       # ...show correct usage
      continue                                          # Go back to the prompt

    if user_input.lower().startswith("tokens "):       # Command: change max tokens
      try:
        max_tokens = int(user_input.split()[1])        # Parse the number after "tokens "
        print(f"Max tokens set to {max_tokens}")       # Confirm the new value
      except (IndexError, ValueError):                 # If parsing failed...
        print("Usage: tokens 20")                      # ...show correct usage
      continue                                          # Go back to the prompt

    response = generate_text(                          # Otherwise: generate a reply
      user_input,                                      # Use the user's text as the prompt
      max_tokens=max_tokens,                           # Reply length
      temperature=temperature,                         # Randomness
      checkpoint=checkpoint,                            # Which model to use
      device=device,                                    # Which device
    )
    print(f"MiniGPT: {response}\n")                    # Print the model's reply


if __name__ == "__main__":                            # Runs only when executed directly
  parser = argparse.ArgumentParser(description="Chat with MiniGPT")  # CLI parser
  parser.add_argument("--checkpoint", default="checkpoints/minigpt.pt")  # --checkpoint
  parser.add_argument("--temperature", type=float, default=0.7)          # --temperature
  parser.add_argument("--max-tokens", type=int, default=15)              # --max-tokens
  parser.add_argument("--device", default=None)                          # --device
  args = parser.parse_args()                          # Parse arguments

  chat(                                                # Start chatting with the parsed settings
    checkpoint=args.checkpoint,
    device=args.device,
    temperature=args.temperature,
    max_tokens=args.max_tokens,
  )
