"""
Dataset utilities for the conditional date generation task.

Each sample is a pair:
  cond  – LongTensor of shape (COND_LEN,)   condition token IDs (no BOS/EOS)
  date  – LongTensor of shape (MAX_DATE_LEN,) char-level date token IDs, PAD-padded

Constants
  COND_LEN     = 6   (day + month + leap + 3 decade-digit tokens)
  MAX_DATE_LEN = 10  (longest date string: "31-12-2200")
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import DateTokenizer

# ── Fixed sequence dimensions ────────────────────────────────────────────────
COND_LEN:     int = 6   # day(1) + month(1) + leap(1) + decade-digits(3)
MAX_DATE_LEN: int = 10  # maximum number of character tokens in a date string

SEED: int = 42


# ─────────────────────────────────────────────────────────────────────────────

class DateDataset(Dataset):
    """
    PyTorch Dataset for the date generation task.

    Parameters
    ----------
    lines     : raw text lines from data.txt
    tokenizer : a DateTokenizer instance
    """

    def __init__(self, lines: List[str], tokenizer: DateTokenizer) -> None:
        self.tokenizer = tokenizer
        self._cond:  List[List[int]] = []
        self._dates: List[List[int]] = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue                          # skip malformed lines

            # ── condition token IDs (6 tokens, no BOS/EOS) ──────────────
            cond_ids: List[int] = [
                tokenizer._token2id[parts[0]],    # day
                tokenizer._token2id[parts[1]],    # month
                tokenizer._token2id[parts[2]],    # leap
            ]
            decade = parts[3][1:-1]               # "[196]" → "196"
            for ch in decade:
                cond_ids.append(tokenizer._token2id[ch])

            # ── date token IDs (character-level) ────────────────────────
            date_ids: List[int] = [tokenizer._token2id[ch] for ch in parts[4]]

            self._cond.append(cond_ids)
            self._dates.append(date_ids)

    def __len__(self) -> int:
        return len(self._cond)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cond_ids  = self._cond[idx]
        date_ids  = self._dates[idx]

        # Pad / truncate date to MAX_DATE_LEN
        pad = DateTokenizer.PAD_ID
        date_padded = (date_ids + [pad] * MAX_DATE_LEN)[:MAX_DATE_LEN]

        return (
            torch.tensor(cond_ids,    dtype=torch.long),
            torch.tensor(date_padded, dtype=torch.long),
        )


# ─────────────────────────────────────────────────────────────────────────────

def load_split(
    data_path: str | Path,
    val_ratio: float = 0.1,
    seed: int = SEED,
) -> Tuple[List[str], List[str]]:
    """
    Read data.txt and return (train_lines, val_lines) after a reproducible
    random shuffle.
    """
    with open(data_path, "r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    rng = random.Random(seed)
    rng.shuffle(lines)

    split = int(len(lines) * (1 - val_ratio))
    return lines[:split], lines[split:]


def make_loaders(
    data_path: str | Path,
    tokenizer: DateTokenizer,
    batch_size: int   = 256,
    val_ratio:  float = 0.1,
    seed:       int   = SEED,
    num_workers: int  = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from data.txt.
    """
    train_lines, val_lines = load_split(data_path, val_ratio, seed)

    train_ds = DateDataset(train_lines, tokenizer)
    val_ds   = DateDataset(val_lines,   tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    print(f"[Dataset]  train={len(train_ds):,}   val={len(val_ds):,}   "
          f"batch={batch_size}")
    return train_loader, val_loader
