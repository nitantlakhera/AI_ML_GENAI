"""
Part 8: Training
----------------
Train MiniGPT on the song lyrics using next-token prediction (cross-entropy loss).

THE TRAINING LOOP IN ONE SENTENCE:
  Show the model a window of words, ask it to predict the next word,
  measure how wrong it is (loss), and nudge the weights to be less wrong.
"""

import argparse                                       # Parse command-line arguments (--epochs, etc.)
from pathlib import Path                              # File paths

import torch                                          # PyTorch core
from torch.optim import AdamW                         # AdamW optimizer (updates weights)

import paths                                          # Locates shared common/ (tokenizer + data)
from dataset import create_dataloader                # Part 1: batched training data
from model import MiniGPT                            # Parts 2-7: the model

# --- Model hyperparameters (as specified in the project brief) ---------------
VOCAB_SIZE = 120                                      # Max vocabulary size
EMBED_SIZE = 64                                       # Embedding dimension
NUM_HEADS = 4                                         # Attention heads per block
NUM_LAYERS = 2                                        # Number of transformer blocks
SEQ_LEN = 32                                          # Context window length
FF_HIDDEN = 256                                       # Feed-forward hidden size

DEFAULT_DATA = str(paths.DATA)                        # Shared training file (common/data/song.txt)
CHECKPOINT_DIR = Path("checkpoints")                  # Where to save the trained model


def train(
  data_path: str = DEFAULT_DATA,                      # Training text path
  epochs: int = 500,                                  # How many full passes over the data
  batch_size: int = 8,                                # Windows per training step
  learning_rate: float = 3e-3,                        # How big each weight update is
  device: str | None = None,                          # "cpu" or "cuda" (auto if None)
) -> None:
  device = device or ("cuda" if torch.cuda.is_available() else "cpu")  # Pick GPU if available
  print(f"Using device: {device}")                    # Tell the user which device is used

  loader, tokenizer = create_dataloader(              # Build the data loader + tokenizer
    data_path,                                        # Path to text
    seq_len=SEQ_LEN,                                  # Window size
    vocab_size=VOCAB_SIZE,                            # Vocab cap
    batch_size=batch_size,                            # Batch size
    shuffle=True,                                     # Shuffle each epoch
  )

  model = MiniGPT(                                     # Create the model with our hyperparameters
    vocab_size=len(tokenizer.word2id),                # Use the ACTUAL vocab learned (<= 120)
    embed_size=EMBED_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    seq_len=SEQ_LEN,
    ff_hidden=FF_HIDDEN,
  ).to(device)                                        # Move model to CPU/GPU

  print(f"Model parameters: {model.count_parameters():,}")  # Show total trainable weights
  print(f"Vocabulary size: {len(tokenizer.word2id)}")       # Show learned vocab size
  print(f"Training samples: {len(loader.dataset)}")         # Show number of windows

  optimizer = AdamW(model.parameters(), lr=learning_rate)   # Optimizer that will update the weights

  CHECKPOINT_DIR.mkdir(exist_ok=True)                 # Make the checkpoints/ folder if missing
  tokenizer.save(CHECKPOINT_DIR / "tokenizer.json")   # Save the vocabulary alongside the model

  model.train()                                       # Put model in "training mode"
  for epoch in range(1, epochs + 1):                  # Loop over epochs (full passes over data)
    total_loss = 0.0                                  # Accumulate loss to average later
    for x, y in loader:                               # Loop over each batch of (input, target)
      x, y = x.to(device), y.to(device)               # Move batch to the device
      optimizer.zero_grad()                           # Reset gradients from the previous step
      loss = model.loss(x, y)                         # Forward pass + compute cross-entropy loss
      loss.backward()                                 # Backpropagation: compute gradients
      optimizer.step()                                # Update weights using the gradients
      total_loss += loss.item()                       # Add this batch's loss to the running total

    avg_loss = total_loss / len(loader)               # Average loss over all batches this epoch
    if epoch % 50 == 0 or epoch == 1:                 # Print progress every 50 epochs (and the first)
      print(f"Epoch {epoch:4d}/{epochs} | Loss: {avg_loss:.4f}")

  checkpoint_path = CHECKPOINT_DIR / "minigpt.pt"     # Path to save the trained weights
  torch.save(                                         # Save model weights + config together
    {
      "model_state_dict": model.state_dict(),         # All learned weights
      "config": {                                     # Config needed to rebuild the model later
        "vocab_size": len(tokenizer.word2id),
        "embed_size": EMBED_SIZE,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "seq_len": SEQ_LEN,
        "ff_hidden": FF_HIDDEN,
      },
    },
    checkpoint_path,                                  # Destination file
  )
  print(f"\nTraining complete! Checkpoint saved to {checkpoint_path}")  # Done!


if __name__ == "__main__":                            # Runs only when executed directly
  parser = argparse.ArgumentParser(description="Train MiniGPT on song lyrics")  # CLI parser
  parser.add_argument("--data", default=DEFAULT_DATA, help="Path to training text")   # --data
  parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")  # --epochs
  parser.add_argument("--batch-size", type=int, default=8, help="Batch size")         # --batch-size
  parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")         # --lr
  parser.add_argument("--device", default=None, help="cpu or cuda")                    # --device
  args = parser.parse_args()                          # Parse the provided arguments

  train(                                              # Kick off training with the parsed args
    data_path=args.data,
    epochs=args.epochs,
    batch_size=args.batch_size,
    learning_rate=args.lr,
    device=args.device,
  )
