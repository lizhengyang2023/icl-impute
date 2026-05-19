from __future__ import annotations

from typing import Iterator, List

import numpy as np
import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    """Ordered sliding windows over multivariate series [T, C]."""

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        step: int = 1,
        start: int = 0,
        end: int | None = None,
    ):
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


def iter_batches(loader, device: torch.device) -> Iterator[torch.Tensor]:
    for batch in loader:
        yield batch.to(device)
