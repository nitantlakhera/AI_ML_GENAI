"""
Part 6: Transformer Block  (TensorFlow / Keras)
-----------------------------------------------
A single GPT-style decoder block:

  x -> LayerNorm -> Multi-Head Attention -> + (residual)
    -> LayerNorm -> Feed-Forward Network  -> + (residual)
    -> output

Built as Keras Layers. Same structure as the PyTorch version.
"""

import tensorflow as tf                               # TensorFlow / Keras

from attention import MultiHeadAttention, create_causal_mask  # Part 4/5 building blocks


class FeedForward(tf.keras.layers.Layer):             # Per-token MLP (Keras Layer)
  """Two-layer MLP with GELU activation (standard in GPT)."""

  def __init__(self, embed_size=64, ff_hidden=256, **kwargs):
    super().__init__(**kwargs)                        # Init Keras Layer
    self.dense1 = tf.keras.layers.Dense(ff_hidden, activation="gelu")  # 64 -> 256 + GELU
    self.dense2 = tf.keras.layers.Dense(embed_size)   # 256 -> 64

  def call(self, x):                                  # Apply MLP to each token
    return self.dense2(self.dense1(x))                # expand -> GELU -> compress


class TransformerBlock(tf.keras.layers.Layer):        # One full decoder block
  def __init__(self, embed_size=64, num_heads=4, ff_hidden=256, seq_len=32, **kwargs):
    super().__init__(**kwargs)                        # Init Keras Layer
    self.seq_len = seq_len                            # Max sequence length
    # LayerNormalization = PyTorch nn.LayerNorm. epsilon matches PyTorch default.
    self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)  # Before attention
    self.attn = MultiHeadAttention(embed_size, num_heads)        # Attention sub-layer
    self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)  # Before feed-forward
    self.ff = FeedForward(embed_size, ff_hidden)                 # Feed-forward sub-layer

  def call(self, x):                                  # Forward pass (Keras call)
    seq_len = tf.shape(x)[1]                           # Current sequence length
    mask = create_causal_mask(seq_len)                # Build causal mask
    x = x + self.attn(self.ln1(x), mask)              # Attention + residual
    x = x + self.ff(self.ln2(x))                      # Feed-forward + residual
    return x                                          # Same shape as input
