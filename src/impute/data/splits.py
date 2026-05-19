from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15


def time_splits(T: int, cfg: SplitConfig) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Return (train, val, test) as (start, end) index ranges along time axis."""
    t1 = int(T * cfg.train_ratio)
    t2 = int(T * (cfg.train_ratio + cfg.val_ratio))
    train = (0, t1)
    val = (t1, t2)
    test = (t2, T)
    return train, val, test
