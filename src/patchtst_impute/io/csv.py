from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_multivariate_csv(path: str | Path) -> Tuple[np.ndarray, List[str]]:
    """Load numeric columns from CSV; skips first column if header name is 'date' (case-insensitive)."""
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
    lower = [h.strip().lower() for h in header]
    skip_first = lower[0] == "date" if lower else False

    usecols = list(range(1, len(header))) if skip_first else list(range(len(header)))

    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=usecols, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data, [header[i] for i in usecols]
