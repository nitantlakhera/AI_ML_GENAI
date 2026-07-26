"""
Part 8: Training  (TensorFlow / Keras)
--------------------------------------
Train MiniGPT on the song lyrics using next-token prediction.

This uses the high-level Keras workflow:
  model.compile(optimizer, loss)  ->  model.fit(dataset, epochs)

That single `fit()` call replaces the manual PyTorch loop of
zero_grad / backward / step (Keras does it internally).
"""

import argparse                                       # Command-line arguments
import json                                           # Save the config as JSON
from pathlib import Path                              # File paths

import tensorflow as tf                               # TensorFlow / Keras

import paths                                          # Locates shared common/ (tokenizer + data)
from dataset import create_dataset                    # Part 1: tf.data pipeline
from model import MiniGPT                             # Parts 2-7: the model

# --- Model hyperparameters (identical to the PyTorch version) ----------------
VOCAB_SIZE = 120                                      # Max vocabulary size
EMBED_SIZE = 64                                       # Embedding dimension
NUM_HEADS = 4                                         # Attention heads per block
NUM_LAYERS = 2                                        # Number of transformer blocks
SEQ_LEN = 32                                          # Context window length
FF_HIDDEN = 256                                       # Feed-forward hidden size

DEFAULT_DATA = str(paths.DATA)                        # Shared training file (common/data/song.txt)
CHECKPOINT_DIR = Path("checkpoints")                  # Where to save the model


def train(
  data_path: str = DEFAULT_DATA,                      # Training text path
  epochs: int = 500,                                  # Full passes over the data
  batch_size: int = 8,                                # Windows per step
  learning_rate: float = 3e-3,                        # Step size
) -> None:
  ds, tokenizer = create_dataset(                     # Build the tf.data pipeline + tokenizer
    data_path,
    seq_len=SEQ_LEN,
    vocab_size=VOCAB_SIZE,
    batch_size=batch_size,
    shuffle=True,
  )

  model = MiniGPT(                                     # Create the Keras model
    vocab_size=len(tokenizer.word2id),                # Actual learned vocab (<= 120)
    embed_size=EMBED_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    seq_len=SEQ_LEN,
    ff_hidden=FF_HIDDEN,
  )

  # Build the model by running one dummy batch through it. Subclassed Keras models
  # create their weights lazily on first call, so this "materializes" them.
  model(tf.zeros((1, SEQ_LEN), dtype=tf.int32))       # One dummy forward pass

  print(f"Model parameters: {model.count_params():,}")  # Total trainable weights
  print(f"Vocabulary size: {len(tokenizer.word2id)}")   # Learned vocab size

  # Keras way: compile with an optimizer + loss, then fit.
  model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate),  # AdamW
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Cross-entropy
  )

  CHECKPOINT_DIR.mkdir(exist_ok=True)                 # Create checkpoints/ folder
  tokenizer.save(CHECKPOINT_DIR / "tokenizer.json")   # Save vocabulary

  # Print progress every 50 epochs (Keras is quiet with verbose=0 + a callback).
  class Progress(tf.keras.callbacks.Callback):        # Custom logging callback
    def on_epoch_end(self, epoch, logs=None):         # Called after each epoch
      if (epoch + 1) % 50 == 0 or epoch == 0:         # Every 50 (and the first)
        print(f"Epoch {epoch + 1:4d}/{epochs} | Loss: {logs['loss']:.4f}")

  model.fit(ds, epochs=epochs, verbose=0, callbacks=[Progress()])  # <-- the training

  # Save the trained weights + the config needed to rebuild the model.
  model.save_weights(str(CHECKPOINT_DIR / "minigpt.weights.h5"))   # Keras weights file
  config = {                                          # Config for rebuilding later
    "vocab_size": len(tokenizer.word2id),
    "embed_size": EMBED_SIZE,
    "num_heads": NUM_HEADS,
    "num_layers": NUM_LAYERS,
    "seq_len": SEQ_LEN,
    "ff_hidden": FF_HIDDEN,
  }
  (CHECKPOINT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
  print(f"\nTraining complete! Weights saved to {CHECKPOINT_DIR / 'minigpt.weights.h5'}")


if __name__ == "__main__":                            # Run only when executed directly
  parser = argparse.ArgumentParser(description="Train MiniGPT (TensorFlow) on song lyrics")
  parser.add_argument("--data", default=DEFAULT_DATA, help="Path to training text")
  parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
  parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
  parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
  args = parser.parse_args()                          # Parse arguments

  train(                                              # Kick off training
    data_path=args.data,
    epochs=args.epochs,
    batch_size=args.batch_size,
    learning_rate=args.lr,
  )
