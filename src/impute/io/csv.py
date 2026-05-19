from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_multivariate_csv(
    path: str | Path,
    encoding: str = "utf-8-sig",
    date_col: str = "date",
    delimiter: str = ","
) -> Tuple[np.ndarray, List[str]]:
    """Load numeric columns from CSV; skips first column if header name is 'date' (case-insensitive).

    Opens the file once with ``encoding`` so header parsing and ``np.loadtxt`` stay
    consistent (passing only a path to ``loadtxt`` uses the OS locale, e.g. gbk on
    Chinese Windows).
    """
    path = Path(path)
    with path.open("r", newline="", encoding=encoding) as f:
        reader = csv.reader(f)
        header = next(reader)
        lower = [h.strip().lower() for h in header]
        skip_first = lower[0] == date_col if lower else False
        usecols = list(range(1, len(header))) if skip_first else list(range(len(header)))
        data = np.loadtxt(f, delimiter=delimiter, usecols=usecols, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data, [header[i] for i in usecols]
