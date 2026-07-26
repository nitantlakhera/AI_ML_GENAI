"""
Part 1 (continued): Dataset  (TensorFlow / Keras version)
---------------------------------------------------------
Creates training samples from tokenized text using `tf.data`.

Each sample is a sliding window of `seq_len` tokens.
Input  = tokens[0 : seq_len]
Target = tokens[1 : seq_len + 1]   (next-token prediction)

PyTorch used torch.utils.data.Dataset/DataLoader.
Here we use tf.data.Dataset — the idiomatic Keras input pipeline.
"""

import tensorflow as tf                               # TensorFlow / Keras backend

import paths  # noqa: F401  (adds the shared common/ folder to the import path)
from tokenizer import build_tokenizer_from_file       # Shared pure-Python tokenizer


def create_dataset(
  text_path: str,                                     # Path to training text
  seq_len: int = 32,                                  # Window length
  vocab_size: int = 120,                              # Vocab cap
  batch_size: int = 8,                                # Windows per training step
  shuffle: bool = True,                               # Shuffle windows each epoch
):
  """Return (tf.data.Dataset, tokenizer). The dataset yields (x, y) batches."""
  tokenizer = build_tokenizer_from_file(text_path, vocab_size=vocab_size)  # Build tokenizer

  with open(text_path, encoding="utf-8") as f:        # Open the text file
    raw_text = f.read()                               # Read the whole song
  tokens = tokenizer.encode(raw_text, add_bos=True, add_eos=True)  # Encode -> IDs

  num_samples = max(0, len(tokens) - seq_len)         # Number of sliding windows

  # Pre-build ALL (x, y) windows as plain lists, then hand them to tf.data.
  # from_tensor_slices is cleanly re-iterable every epoch (no generator quirks).
  xs, ys = [], []                                     # Inputs and targets
  for i in range(num_samples):                        # Slide over the token stream
    chunk = tokens[i : i + seq_len + 1]               # seq_len+1 tokens
    xs.append(chunk[:-1])                             # Input  = all but last
    ys.append(chunk[1:])                              # Target = shifted by 1

  x_tensor = tf.constant(xs, dtype=tf.int32)          # (num_samples, seq_len)
  y_tensor = tf.constant(ys, dtype=tf.int32)          # (num_samples, seq_len)

  ds = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))  # One slice per window
  if shuffle:                                         # Shuffle for better training
    ds = ds.shuffle(buffer_size=max(1, num_samples))  # Shuffle all windows
  ds = ds.batch(batch_size)                           # Group into batches
  ds = ds.prefetch(tf.data.AUTOTUNE)                  # Overlap data prep with training

  return ds, tokenizer                                # Return pipeline + tokenizer
