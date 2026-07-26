"""
Part 1: Tokenizer
-----------------
Converts raw text into integer token IDs and back.

We use a simple word-level tokenizer:
  - Split text on whitespace
  - Build a vocabulary from unique words
  - Reserve special tokens: <PAD>, <UNK>, <BOS>, <EOS>

WHY: A neural network cannot read words. It only understands numbers.
The tokenizer is the "translator" between human text and model numbers.
"""

import json                       # Used to save/load the vocabulary as a .json file
import re                         # Regular expressions, used to clean/normalize text
from pathlib import Path          # Object-oriented file paths (works on Windows/Linux/Mac)

# --- Special tokens -----------------------------------------------------------
# These 4 tokens have fixed meanings and always occupy the first 4 IDs (0-3).
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]  # Ordered list of special tokens
PAD_ID = 0                        # <PAD>: filler used to make sequences equal length
UNK_ID = 1                        # <UNK>: "unknown" — any word not in the vocabulary
BOS_ID = 2                        # <BOS>: "beginning of sequence" marker
EOS_ID = 3                        # <EOS>: "end of sequence" marker


class Tokenizer:                                      # The tokenizer class
    def __init__(self, vocab_size: int = 120):        # Constructor; default vocab cap = 120
        self.vocab_size = vocab_size                  # Store the maximum vocabulary size
        self.word2id: dict[str, int] = {}             # Map: word (str) -> token ID (int)
        self.id2word: dict[int, str] = {}             # Reverse map: token ID (int) -> word (str)

    def _normalize(self, text: str) -> str:           # Clean raw text before splitting
        text = text.lower().strip()                   # Lowercase + trim leading/trailing spaces
        text = re.sub(r"[^\w\s']", " ", text)         # Replace punctuation (keep letters/digits/spaces/')
        text = re.sub(r"\s+", " ", text)              # Collapse multiple spaces into one space
        return text                                   # Return the cleaned string

    def _tokenize_words(self, text: str) -> list[str]:  # Turn a string into a list of words
        return self._normalize(text).split()          # Normalize, then split on whitespace

    def build_vocab(self, texts: list[str]) -> None:  # Learn the vocabulary from documents
        """Build vocabulary from a list of text documents."""
        word_freq: dict[str, int] = {}                # Dictionary counting how often each word appears
        for text in texts:                            # Loop over each document
            for word in self._tokenize_words(text):   # Loop over each word in the document
                word_freq[word] = word_freq.get(word, 0) + 1  # Increment that word's count

        # Sort words by frequency (most common first); ties broken alphabetically.
        sorted_words = sorted(word_freq.keys(), key=lambda w: (-word_freq[w], w))

        # Assign IDs 0-3 to the special tokens first.
        self.word2id = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}  # word -> id
        self.id2word = {i: tok for tok, i in self.word2id.items()}       # id -> word

        remaining = self.vocab_size - len(SPECIAL_TOKENS)  # How many normal words we can still add
        for word in sorted_words[:remaining]:         # Take the top-N most frequent words
            idx = len(self.word2id)                   # Next free ID = current dictionary size
            self.word2id[word] = idx                  # Record word -> id
            self.id2word[idx] = word                  # Record id -> word

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        # Convert a text string into a list of integer token IDs.
        words = self._tokenize_words(text)            # Split text into words
        ids = []                                      # Output list of IDs
        if add_bos:                                   # Optionally prepend <BOS>
            ids.append(BOS_ID)
        for word in words:                            # For every word...
            ids.append(self.word2id.get(word, UNK_ID))  # ...look up its ID, or <UNK> if unknown
        if add_eos:                                   # Optionally append <EOS>
            ids.append(EOS_ID)
        return ids                                    # Return the list of token IDs

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        # Convert a list of token IDs back into a readable string.
        words = []                                    # Output list of words
        special = set(SPECIAL_TOKENS) if skip_special else set()  # Which tokens to hide
        for idx in ids:                               # For each token ID...
            word = self.id2word.get(idx, "<UNK>")     # ...find its word (or <UNK>)
            if word in special:                       # Skip special tokens if requested
                continue
            words.append(word)                        # Keep the real word
        return " ".join(words)                        # Join words back into a sentence

    def save(self, path: str | Path) -> None:         # Save vocabulary to disk (JSON)
        path = Path(path)                             # Convert to a Path object
        data = {                                      # Bundle everything we need to reload
            "vocab_size": self.vocab_size,            # Store the vocab cap
            "word2id": self.word2id,                  # Store the word -> id map
            "id2word": {str(k): v for k, v in self.id2word.items()},  # JSON keys must be strings
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")  # Write pretty JSON

    @classmethod                                      # Alternate constructor (loads from file)
    def load(cls, path: str | Path) -> "Tokenizer":
        path = Path(path)                             # Convert to a Path object
        data = json.loads(path.read_text(encoding="utf-8"))  # Read + parse the JSON file
        tok = cls(vocab_size=data["vocab_size"])      # Create a new Tokenizer instance
        tok.word2id = data["word2id"]                 # Restore word -> id map
        tok.id2word = {int(k): v for k, v in data["id2word"].items()}  # Restore id -> word (keys back to int)
        return tok                                    # Return the ready-to-use tokenizer


def build_tokenizer_from_file(text_path: str | Path, vocab_size: int = 120) -> Tokenizer:
    # Convenience helper: read a text file and build a tokenizer from it.
    text = Path(text_path).read_text(encoding="utf-8")  # Read the whole file into a string
    tokenizer = Tokenizer(vocab_size=vocab_size)      # Create an empty tokenizer
    tokenizer.build_vocab([text])                     # Build vocab from the single document
    return tokenizer                                  # Return the built tokenizer


if __name__ == "__main__":                            # Runs only when executed directly (python tokenizer.py)
    _song = Path(__file__).resolve().parent / "data" / "song.txt"  # common/data/song.txt
    tok = build_tokenizer_from_file(_song)            # Build a tokenizer from the song
    print(f"Vocabulary size: {len(tok.word2id)}")     # Show how many tokens were learned
    sample = "twinkle twinkle little star"            # A sample sentence to test
    ids = tok.encode(sample)                           # Encode it to IDs
    print(f"Encode: {sample!r} -> {ids}")             # Print the IDs
    print(f"Decode: {tok.decode(ids)!r}")             # Decode back and print (should match input)
