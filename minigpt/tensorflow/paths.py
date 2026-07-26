"""
paths.py — locate the shared `common/` folder.

Both the PyTorch and TensorFlow versions share ONE tokenizer and ONE dataset,
which live in `../common/`. Importing this module:
  1. finds the `common/` folder (searching upward from this file),
  2. adds it to sys.path so `import tokenizer` works, and
  3. exposes COMMON (the folder) and DATA (the song.txt path).

This makes the code work no matter what directory you run it from, and also
inside Docker (where common/ is copied next to the framework folder).
"""

import sys                                            # To modify the import path
from pathlib import Path                              # Cross-platform paths


def _find_common() -> Path:                           # Search upward for common/
  here = Path(__file__).resolve()                     # Absolute path to this file
  for parent in here.parents:                         # Walk up: tensorflow/ -> repo root -> ...
    candidate = parent / "common"                     # Look for a sibling 'common' folder
    if (candidate / "tokenizer.py").exists():         # Confirm it's the right one
      return candidate                                # Found it
  raise RuntimeError("Could not locate the shared 'common/' folder")  # Should never happen


COMMON = _find_common()                               # The common/ directory
if str(COMMON) not in sys.path:                       # Only add once
  sys.path.insert(0, str(COMMON))                     # Make common/ importable

DATA = COMMON / "data" / "song.txt"                   # Shared training data file
