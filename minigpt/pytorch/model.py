"""
Parts 2, 3 & 7: Embeddings, Positional Encoding, and Full GPT Model
---------------------------------------------------------------------
Part 2 — Token Embeddings:  each token ID -> dense vector (learned lookup table)
Part 3 — Positional Embeddings: add position info so order matters
Part 7 — Stack transformer blocks into MiniGPT

BIG PICTURE (data shapes):
  token IDs (batch, seq)  ->  embeddings (batch, seq, 64)
  -> transformer blocks (batch, seq, 64)  ->  logits (batch, seq, vocab)
"""

import torch                                          # PyTorch core
import torch.nn as nn                                 # Neural-network layers
import torch.nn.functional as F                       # Functional ops (cross_entropy, softmax)

from transformer import TransformerBlock              # Part 6 building block


class MiniGPT(nn.Module):                             # The full model
  def __init__(
    self,
    vocab_size: int = 120,                            # Number of tokens the model knows
    embed_size: int = 64,                             # Size of each token's vector
    num_heads: int = 4,                               # Attention heads per block
    num_layers: int = 2,                              # How many transformer blocks to stack
    seq_len: int = 32,                                # Maximum context length
    ff_hidden: int = 256,                             # Feed-forward hidden size
  ):
    super().__init__()                                # Initialize parent nn.Module
    self.vocab_size = vocab_size                      # Save for later (loss, generation)
    self.embed_size = embed_size                      # Save embedding size
    self.seq_len = seq_len                            # Save max sequence length

    # Part 2: Token embeddings — a lookup table with one 64-dim vector per token.
    self.token_embedding = nn.Embedding(vocab_size, embed_size)

    # Part 3: Learned positional embeddings — one 64-dim vector per position (0..seq_len-1).
    self.pos_embedding = nn.Embedding(seq_len, embed_size)

    # Part 7: Stack of transformer blocks (the "depth" of the network).
    self.blocks = nn.ModuleList(                      # ModuleList registers each block as a sub-module
      [
        TransformerBlock(embed_size, num_heads, ff_hidden, seq_len)  # Build one block...
        for _ in range(num_layers)                    # ...repeated num_layers times
      ]
    )

    self.ln_f = nn.LayerNorm(embed_size)             # Final LayerNorm before the output head
    self.head = nn.Linear(embed_size, vocab_size, bias=False)  # Maps 64-dim vector -> vocab scores

    self._init_weights()                              # Initialize all weights sensibly

  def _init_weights(self) -> None:                    # Good initial weights help training start well
    for module in self.modules():                     # Loop over every sub-module
      if isinstance(module, nn.Linear):               # For linear layers...
        nn.init.normal_(module.weight, mean=0.0, std=0.02)  # ...small random weights
        if module.bias is not None:                   # If the layer has a bias...
          nn.init.zeros_(module.bias)                 # ...start it at zero
      elif isinstance(module, nn.Embedding):          # For embedding tables...
        nn.init.normal_(module.weight, mean=0.0, std=0.02)  # ...small random weights too

  def forward(self, idx: torch.Tensor) -> torch.Tensor:
    """
    idx: (batch, seq_len) token IDs
    returns logits: (batch, seq_len, vocab_size)
    """
    batch, seq_len = idx.shape                        # Unpack batch size and sequence length
    assert seq_len <= self.seq_len, f"Sequence length {seq_len} exceeds max {self.seq_len}"  # Safety check

    tok_emb = self.token_embedding(idx)               # Look up token vectors: (batch, seq, 64)
    positions = torch.arange(seq_len, device=idx.device).unsqueeze(0)  # [0,1,...,seq-1] shape (1, seq)
    pos_emb = self.pos_embedding(positions)           # Look up position vectors: (1, seq, 64)

    x = tok_emb + pos_emb                             # Combine "what word" + "which position"

    for block in self.blocks:                         # Pass through each transformer block in order
      x = block(x)                                    # Each block refines the representation

    x = self.ln_f(x)                                  # Final normalization
    return self.head(x)                               # Project to vocab-sized scores (logits)

  def loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # Part 8 helper
    logits = self.forward(idx)                        # Run the forward pass to get predictions
    # cross_entropy compares predicted token distribution vs. the true next token.
    return F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))  # Flatten then compare

  @torch.no_grad()                                    # Disable gradient tracking (we're only predicting)
  def generate(
    self,
    idx: torch.Tensor,                                # Starting token IDs: (batch, seq)
    max_new_tokens: int,                              # How many new tokens to produce
    temperature: float = 1.0,                         # >1 = more random, <1 = more focused
    top_k: int | None = None,                         # Keep only the k most likely tokens
  ) -> torch.Tensor:
    for _ in range(max_new_tokens):                   # Generate one token at a time
      idx_cond = idx[:, -self.seq_len :]              # Keep only the last seq_len tokens (context window)
      logits = self.forward(idx_cond)[:, -1, :]       # Predict; take logits for the LAST position only

      if temperature != 1.0:                          # Apply temperature scaling if requested
        logits = logits / temperature                 # Divide logits: lower temp -> sharper distribution

      if top_k is not None:                           # Apply top-k filtering if requested
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # Find the top-k logit values
        logits[logits < values[:, [-1]]] = float("-inf")  # Zero-out everything below the k-th value

      probs = F.softmax(logits, dim=-1)               # Convert logits to probabilities
      next_token = torch.multinomial(probs, num_samples=1)  # Randomly sample the next token
      idx = torch.cat([idx, next_token], dim=1)       # Append it to the running sequence

    return idx                                        # Return the full sequence (prompt + generated)

  def count_parameters(self) -> int:                  # Utility: count trainable weights
    return sum(p.numel() for p in self.parameters() if p.requires_grad)  # Sum sizes of all params
