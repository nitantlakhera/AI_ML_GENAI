"""
Parts 2, 3 & 7: Embeddings, Positional Encoding, Full GPT  (TensorFlow / Keras)
-------------------------------------------------------------------------------
Part 2 — Token Embeddings:  token ID -> dense vector (learned lookup table)
Part 3 — Positional Embeddings: add position info so order matters
Part 7 — Stack transformer blocks into MiniGPT

Built as a tf.keras.Model subclass — the Keras equivalent of nn.Module.
"""

import tensorflow as tf                               # TensorFlow / Keras

from transformer import TransformerBlock              # Part 6 building block


class MiniGPT(tf.keras.Model):                        # Keras Model subclass (like nn.Module)
  def __init__(
    self,
    vocab_size=120,                                   # Tokens the model knows
    embed_size=64,                                    # Size of each token vector
    num_heads=4,                                      # Attention heads per block
    num_layers=2,                                     # Number of transformer blocks
    seq_len=32,                                       # Max context length
    ff_hidden=256,                                    # Feed-forward hidden size
    **kwargs,
  ):
    super().__init__(**kwargs)                        # Init Keras Model
    self.vocab_size = vocab_size                      # Save config
    self.embed_size = embed_size
    self.seq_len = seq_len

    # Part 2: Token embeddings (Keras Embedding = PyTorch nn.Embedding).
    self.token_embedding = tf.keras.layers.Embedding(vocab_size, embed_size)

    # Part 3: Learned positional embeddings.
    self.pos_embedding = tf.keras.layers.Embedding(seq_len, embed_size)

    # Part 7: Stack of transformer blocks.
    self.blocks = [
      TransformerBlock(embed_size, num_heads, ff_hidden, seq_len)  # One block...
      for _ in range(num_layers)                      # ...repeated num_layers times
    ]

    self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-5)  # Final LayerNorm
    self.head = tf.keras.layers.Dense(vocab_size, use_bias=False)  # 64 -> vocab logits

  def call(self, idx, training=False):                # Forward pass (Keras call)
    """
    idx: (batch, seq_len) int token IDs
    returns logits: (batch, seq_len, vocab_size)
    """
    seq_len = tf.shape(idx)[1]                         # Current sequence length

    tok_emb = self.token_embedding(idx)               # (batch, seq, 64)
    positions = tf.range(seq_len)                      # [0, 1, ..., seq-1]
    pos_emb = self.pos_embedding(positions)            # (seq, 64)

    x = tok_emb + pos_emb                             # Combine token + position (broadcast)

    for block in self.blocks:                         # Pass through each block
      x = block(x)                                    # Refine representation

    x = self.ln_f(x)                                  # Final normalization
    return self.head(x)                               # Logits over vocabulary

  def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    """Autoregressive generation. idx: (batch, seq) int tensor."""
    for _ in range(max_new_tokens):                   # One token at a time
      idx_cond = idx[:, -self.seq_len:]               # Keep last seq_len tokens (context)
      logits = self(idx_cond, training=False)         # Forward pass
      logits = logits[:, -1, :]                       # Take last position's logits

      if temperature != 1.0:                          # Temperature scaling
        logits = logits / temperature

      if top_k is not None:                           # Top-k filtering
        k = tf.minimum(top_k, tf.shape(logits)[-1])   # Clamp k to vocab size
        values, _ = tf.math.top_k(logits, k=k)        # Top-k logit values
        min_keep = values[:, -1:]                     # The k-th largest per row
        logits = tf.where(logits < min_keep,          # Below threshold?
                          tf.fill(tf.shape(logits), -1e9),  # -> -inf
                          logits)                      # else keep

      # Sample the next token from the probability distribution.
      next_token = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)
      idx = tf.concat([idx, next_token], axis=1)      # Append to the sequence

    return idx                                        # Full sequence (prompt + generated)
