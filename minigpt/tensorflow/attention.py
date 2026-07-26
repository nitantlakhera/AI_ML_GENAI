"""
Parts 4 & 5: Self-Attention and Multi-Head Attention  (TensorFlow / Keras)
--------------------------------------------------------------------------
Same math as the PyTorch version:
  Q (Query)  = "what am I looking for?"
  K (Key)    = "what do I contain?"
  V (Value)  = "what information do I pass on?"

We build the multi-head attention as a Keras Layer (tf.keras.layers.Layer)
using Dense layers for the Q/K/V/O projections.
"""

import tensorflow as tf                               # TensorFlow / Keras


def scaled_dot_product_attention(q, k, v, mask=None): # Core attention formula
  """
  Attention(Q, K, V) = softmax(Q @ Kᵀ / sqrt(d_k)) @ V
  q, k, v shape: (batch, heads, seq_len, head_dim)
  mask:          (1, 1, seq_len, seq_len) with 1.0 = keep, 0.0 = block
  """
  d_k = tf.cast(tf.shape(q)[-1], tf.float32)          # head_dim as a float
  scores = tf.matmul(q, k, transpose_b=True)          # Q @ Kᵀ  (transpose last 2 dims of k)
  scores = scores / tf.math.sqrt(d_k)                 # Scale by √d_k

  if mask is not None:                                # Apply causal mask if given
    # Additive mask: blocked positions get a huge negative number (-> ~0 after softmax).
    scores += (1.0 - mask) * -1e9                     # Keras-style additive masking

  weights = tf.nn.softmax(scores, axis=-1)            # Attention probabilities
  return tf.matmul(weights, v)                        # Weighted sum of Values


def create_causal_mask(seq_len):                      # "No peeking at the future" mask
  """Lower-triangular matrix of 1s; shape (1, 1, seq_len, seq_len)."""
  mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)  # Keep lower triangle
  return mask[tf.newaxis, tf.newaxis, :, :]           # Add batch & head dims


class MultiHeadAttention(tf.keras.layers.Layer):      # Keras Layer subclass
  def __init__(self, embed_size=64, num_heads=4, **kwargs):
    super().__init__(**kwargs)                        # Initialize the Keras Layer
    assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"

    self.embed_size = embed_size                      # Full embedding dim (e.g. 64)
    self.num_heads = num_heads                        # Number of heads (e.g. 4)
    self.head_dim = embed_size // num_heads           # Dim per head (e.g. 16)

    # Keras Dense layers = PyTorch nn.Linear (use_bias=False to match).
    self.W_q = tf.keras.layers.Dense(embed_size, use_bias=False)  # -> Queries
    self.W_k = tf.keras.layers.Dense(embed_size, use_bias=False)  # -> Keys
    self.W_v = tf.keras.layers.Dense(embed_size, use_bias=False)  # -> Values
    self.W_o = tf.keras.layers.Dense(embed_size, use_bias=False)  # -> Output mix

  def _split_heads(self, x):                          # (batch, seq, embed) -> (batch, heads, seq, head_dim)
    batch = tf.shape(x)[0]                             # Batch size
    seq_len = tf.shape(x)[1]                           # Sequence length
    x = tf.reshape(x, (batch, seq_len, self.num_heads, self.head_dim))  # Split embed dim
    return tf.transpose(x, perm=[0, 2, 1, 3])         # Move heads before seq

  def _merge_heads(self, x):                          # Reverse of _split_heads
    batch = tf.shape(x)[0]                             # Batch size
    seq_len = tf.shape(x)[2]                           # Sequence length
    x = tf.transpose(x, perm=[0, 2, 1, 3])            # Move seq back before heads
    return tf.reshape(x, (batch, seq_len, self.embed_size))  # Flatten heads

  def call(self, x, mask=None):                       # Keras uses call() (PyTorch uses forward())
    q = self._split_heads(self.W_q(x))                # Project -> Q, split heads
    k = self._split_heads(self.W_k(x))                # Project -> K, split heads
    v = self._split_heads(self.W_v(x))                # Project -> V, split heads

    attn_out = scaled_dot_product_attention(q, k, v, mask)  # Attention for all heads
    return self.W_o(self._merge_heads(attn_out))      # Merge heads + output projection
