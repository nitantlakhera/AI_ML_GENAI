"""
Part 1 (continued): Dataset
---------------------------
Creates training samples from tokenized text.

Each sample is a sliding window of `seq_len` tokens.
Input  = tokens[0 : seq_len]
Target = tokens[1 : seq_len + 1]   (next-token prediction)

WHY: GPT learns by predicting the NEXT word. So for every window of words,
the "answer" (target) is the same window shifted one step to the right.
"""

import torch                                          # PyTorch: tensors + autograd
from torch.utils.data import Dataset                  # Base class for custom datasets

import paths  # noqa: F401  (adds the shared common/ folder to the import path)
from tokenizer import Tokenizer, build_tokenizer_from_file  # Our Part-1 tokenizer


class SongDataset(Dataset):                           # Custom dataset for the song lyrics
  def __init__(self, text_path: str, seq_len: int = 32, vocab_size: int = 120):
    self.seq_len = seq_len                            # How many tokens per training window
    self.tokenizer = build_tokenizer_from_file(text_path, vocab_size=vocab_size)  # Build tokenizer
    self.tokens = self.tokenizer.encode(              # Encode the WHOLE song into one long list of IDs
      open(text_path, encoding="utf-8").read(),       # Read the entire file as one string
      add_bos=True,                                   # Put <BOS> at the very start
      add_eos=True,                                   # Put <EOS> at the very end
    )
    self.num_samples = max(0, len(self.tokens) - seq_len)  # Number of sliding windows we can make

  def __len__(self) -> int:                           # Required: how many samples exist
    return self.num_samples                           # Return the count computed above

  def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # Required: get one sample
    chunk = self.tokens[idx : idx + self.seq_len + 1]  # Grab seq_len+1 tokens starting at idx
    x = torch.tensor(chunk[:-1], dtype=torch.long)    # Input  = all but the last token
    y = torch.tensor(chunk[1:], dtype=torch.long)     # Target = all but the first (shifted by 1)
    return x, y                                        # Return (input, target) pair

  @property
  def vocab_size(self) -> int:                        # Convenience: actual vocabulary size
    return len(self.tokenizer.word2id)                # Number of tokens the tokenizer learned


def create_dataloader(                                # Helper: build a batching DataLoader
  text_path: str,                                     # Path to the training text
  seq_len: int = 32,                                  # Window length
  vocab_size: int = 120,                              # Vocab cap
  batch_size: int = 8,                                # How many windows per training step
  shuffle: bool = True,                               # Shuffle windows each epoch (good for training)
):
  from torch.utils.data import DataLoader             # Import here to keep top of file clean

  dataset = SongDataset(text_path, seq_len=seq_len, vocab_size=vocab_size)  # Build the dataset
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)      # Wrap it in a DataLoader
  return loader, dataset.tokenizer                    # Return both the loader and the tokenizer
