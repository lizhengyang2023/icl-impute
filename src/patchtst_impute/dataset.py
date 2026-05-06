from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15


def load_multivariate_csv(path: str | Path) -> Tuple[np.ndarray, List[str]]:
    """Load numeric columns from CSV; skips first column if header name is 'date' (case-insensitive)."""
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
    lower = [h.strip().lower() for h in header]
    skip_first = lower[0] == "date" if lower else False

    cols = header[1:] if skip_first else header
    usecols = list(range(1, len(header))) if skip_first else list(range(len(header)))

    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=usecols, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data, [header[i] for i in usecols]


class SlidingWindowDataset(Dataset):
    """Ordered sliding windows over multivariate series [T, C]."""

    def __init__(self, data: np.ndarray, seq_len: int, step: int = 1, start: int = 0, end: int | None = None):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.step = max(1, step)
        T = data.shape[0]
        if end is None:
            end = T
        self.start = max(0, start)
        self.end = min(T, end)
        last = self.end - seq_len
        if last < self.start:
            self.indices: List[int] = []
        else:
            self.indices = list(range(self.start, last + 1, self.step))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        s = self.indices[idx]
        w = self.data[s : s + self.seq_len]
        return torch.from_numpy(w)


def time_splits(T: int, cfg: SplitConfig) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    t1 = int(T * cfg.train_ratio)
    t2 = int(T * (cfg.train_ratio + cfg.val_ratio))
    train = (0, t1)
    val = (t1, t2)
    test = (t2, T)
    return train, val, test


def iter_batches(loader, device: torch.device) -> Iterator[torch.Tensor]:
    for batch in loader:
        yield batch.to(device)
