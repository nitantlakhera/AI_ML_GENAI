"""
Parts 4 & 5: Self-Attention and Multi-Head Attention
----------------------------------------------------
Self-attention lets each token "look at" other tokens and decide
which ones are important.

Multi-head attention runs several attention operations in parallel,
each learning different relationships (e.g. rhyme, repetition, context).

KEY IDEA: For each word we build 3 vectors:
  Q (Query)  = "what am I looking for?"
  K (Key)    = "what do I contain?"
  V (Value)  = "what information do I pass on?"
Attention compares every Query with every Key to decide how much of each Value to keep.
"""

import math                                           # For sqrt() used to scale the scores

import torch                                          # PyTorch core
import torch.nn as nn                                 # Neural-network layers (Linear, etc.)
import torch.nn.functional as F                       # Functional ops (softmax, etc.)


def scaled_dot_product_attention(                     # The core attention formula
  q: torch.Tensor,                                    # Queries: (batch, heads, seq_len, head_dim)
  k: torch.Tensor,                                    # Keys:    (batch, heads, seq_len, head_dim)
  v: torch.Tensor,                                    # Values:  (batch, heads, seq_len, head_dim)
  mask: torch.Tensor | None = None,                   # Optional causal mask
) -> torch.Tensor:
  """
  Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

  q, k, v shape: (batch, heads, seq_len, head_dim)
  mask:          (1, 1, seq_len, seq_len) — True = keep, False = block
  """
  d_k = q.size(-1)                                    # head_dim: size of each head's vector
  scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # Q·Kᵀ scaled by √d_k
  # scores[i][j] = how much token i should attend to token j (before masking/softmax)

  if mask is not None:                                # If a causal mask was provided...
    scores = scores.masked_fill(~mask, float("-inf")) # ...set blocked positions to -inf (0 after softmax)

  weights = F.softmax(scores, dim=-1)                 # Turn scores into probabilities that sum to 1
  return torch.matmul(weights, v)                     # Weighted sum of Values = attention output


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
  """
  Causal (autoregressive) mask — token i can only attend to tokens <= i.
  Prevents the model from "cheating" by looking at future tokens during training.
  """
  mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))  # Lower-triangular True
  return mask.unsqueeze(0).unsqueeze(0)              # Add batch & head dims -> (1, 1, seq_len, seq_len)


class MultiHeadAttention(nn.Module):                  # Part 5: runs several attention "heads" at once
  def __init__(self, embed_size: int = 64, num_heads: int = 4):
    super().__init__()                                # Initialize the parent nn.Module
    assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"  # Must divide evenly

    self.embed_size = embed_size                      # Full embedding dimension (e.g. 64)
    self.num_heads = num_heads                        # Number of parallel heads (e.g. 4)
    self.head_dim = embed_size // num_heads           # Dimension per head (e.g. 64/4 = 16)

    self.W_q = nn.Linear(embed_size, embed_size, bias=False)  # Learns to produce Queries
    self.W_k = nn.Linear(embed_size, embed_size, bias=False)  # Learns to produce Keys
    self.W_v = nn.Linear(embed_size, embed_size, bias=False)  # Learns to produce Values
    self.W_o = nn.Linear(embed_size, embed_size, bias=False)  # Mixes heads back together (output)

  def _split_heads(self, x: torch.Tensor) -> torch.Tensor:  # (batch, seq, embed) -> (batch, heads, seq, head_dim)
    batch, seq_len, _ = x.shape                       # Unpack batch size and sequence length
    x = x.view(batch, seq_len, self.num_heads, self.head_dim)  # Split embed dim into heads
    return x.transpose(1, 2)                          # Move heads before seq: (batch, heads, seq, head_dim)

  def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:  # Reverse of _split_heads
    batch, _, seq_len, _ = x.shape                    # Unpack batch and seq length
    x = x.transpose(1, 2).contiguous()                # Move seq back before heads: (batch, seq, heads, head_dim)
    return x.view(batch, seq_len, self.embed_size)    # Flatten heads back into one embed vector

  def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    q = self._split_heads(self.W_q(x))                # Project input -> Queries, then split into heads
    k = self._split_heads(self.W_k(x))                # Project input -> Keys, then split into heads
    v = self._split_heads(self.W_v(x))                # Project input -> Values, then split into heads

    attn_out = scaled_dot_product_attention(q, k, v, mask)  # Run attention for all heads at once
    return self.W_o(self._merge_heads(attn_out))      # Merge heads, then final output projection
