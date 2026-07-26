"""
Part 6: Transformer Block
-------------------------
A single GPT-style decoder block:

  x -> LayerNorm -> Multi-Head Attention -> + (residual)
    -> LayerNorm -> Feed-Forward Network  -> + (residual)
    -> output

Residual connections help gradients flow during backpropagation.
Layer normalization stabilizes training.

ANALOGY:
  - Attention = tokens "talk to each other" and share context.
  - Feed-Forward = each token "thinks on its own" about what it learned.
"""

import torch                                          # PyTorch core
import torch.nn as nn                                 # Neural-network layers

from attention import MultiHeadAttention, create_causal_mask  # Part 4/5 building blocks


class FeedForward(nn.Module):                         # The per-token "thinking" network (an MLP)
  """Two-layer MLP with GELU activation (standard in GPT)."""

  def __init__(self, embed_size: int = 64, ff_hidden: int = 256):
    super().__init__()                                # Initialize parent nn.Module
    self.net = nn.Sequential(                         # Stack layers in order:
      nn.Linear(embed_size, ff_hidden),               #   64 -> 256 (expand: more "neurons" to think with)
      nn.GELU(),                                       #   Non-linear activation (adds expressive power)
      nn.Linear(ff_hidden, embed_size),               #   256 -> 64 (compress back to embedding size)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor: # Apply the MLP to every token independently
    return self.net(x)                                # Run input through the Sequential stack


class TransformerBlock(nn.Module):                    # One full decoder block (attention + FF)
  def __init__(
    self,
    embed_size: int = 64,                             # Embedding dimension
    num_heads: int = 4,                               # Number of attention heads
    ff_hidden: int = 256,                             # Hidden size of the feed-forward network
    seq_len: int = 32,                                # Max sequence length (for the mask)
  ):
    super().__init__()                                # Initialize parent nn.Module
    self.seq_len = seq_len                            # Store max sequence length
    self.ln1 = nn.LayerNorm(embed_size)              # LayerNorm applied BEFORE attention (Pre-LN)
    self.attn = MultiHeadAttention(embed_size, num_heads)  # The multi-head attention sub-layer
    self.ln2 = nn.LayerNorm(embed_size)              # LayerNorm applied BEFORE the feed-forward
    self.ff = FeedForward(embed_size, ff_hidden)     # The feed-forward sub-layer

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    mask = create_causal_mask(x.size(1), x.device)   # Build the "no peeking at future" mask
    x = x + self.attn(self.ln1(x), mask)             # Attention + residual (add input back in)
    x = x + self.ff(self.ln2(x))                     # Feed-forward + residual (add input back in)
    return x                                          # Output has same shape as input
