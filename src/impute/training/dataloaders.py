from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from torch.utils.data import DataLoader

from impute.data import SplitConfig, SlidingWindowDataset, time_splits
from impute.io import load_multivariate_csv


def resolve_data_path(path: str | Path, repo_root: Path) -> Path:
    """Accept absolute path, cwd-relative path, or path relative to repo root."""
    p = Path(path)
    if p.is_file():
        return p
    cand = repo_root / path
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"data_path not found: {path}")


def build_time_series_loaders(
    data_path: str | Path,
    repo_root: Path,
    *,
    seq_len: int,
    step: int,
    batch_size: int,
    num_workers: int,
    split: SplitConfig | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, List[str]]:
    """
    Load multivariate CSV, apply temporal splits, return train/val/test DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, n_vars, column_names
    """
    split = split or SplitConfig()
    resolved = resolve_data_path(data_path, repo_root)
    raw, col_names = load_multivariate_csv(str(resolved))
    T, n_vars = raw.shape
    train_r, val_r, test_r = time_splits(T, split)

    train_ds = SlidingWindowDataset(raw, seq_len, step=step, start=train_r[0], end=train_r[1])
    val_ds = SlidingWindowDataset(raw, seq_len, step=step, start=val_r[0], end=val_r[1])
    test_ds = SlidingWindowDataset(raw, seq_len, step=step, start=test_r[0], end=test_r[1])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, n_vars, col_names
