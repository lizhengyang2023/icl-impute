"""
Train PatchTST-style imputer with random point masking (self-supervised on fully observed CSV).
Run from repo root:
  python -m patchtst_impute.train --data_path test_data/ETT-small/ETTh1.csv
(add src to PYTHONPATH, or run from src directory)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Allow `python src/patchtst_impute/train.py` and `python -m patchtst_impute.train`
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from patchtst_impute.dataset import SplitConfig, SlidingWindowDataset, load_multivariate_csv, time_splits
from patchtst_impute.model import PatchTSTImputer


def make_train_mask(
    x: torch.Tensor,
    mask_ratio: float,
    min_observed: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    x: [B,L,C] fully observed windows.
    Returns obs_mask [B,L,C] with 1 where the model may read the value, 0 where imputed at train time.
    Ensures each (b,c) has at least `min_observed` ones when L allows.
    """
    B, L, C = x.shape
    device = x.device
    obs = torch.ones((B, L, C), device=device, dtype=torch.float32)
    n_hide = int(round(L * mask_ratio))
    n_hide = max(0, min(L - min_observed, n_hide))
    if n_hide == 0:
        return obs
    for b in range(B):
        for c in range(C):
            idx = torch.randperm(L, device=device, generator=generator)[:n_hide]
            obs[b, idx, c] = 0.0
    return obs


def masked_mse(pred: torch.Tensor, target: torch.Tensor, miss: torch.Tensor) -> torch.Tensor:
    """miss: 1 on positions to score (held-out / missing)."""
    d = (pred - target).pow(2) * miss
    den = miss.sum().clamp(min=1.0)
    return d.sum() / den


@torch.no_grad()
def evaluate(model: PatchTSTImputer, loader: DataLoader, device: torch.device, mask_ratio: float) -> float:
    model.eval()
    total = 0.0
    n = 0
    for x in loader:
        x = x.to(device)
        B, L, C = x.shape
        obs = make_train_mask(x, mask_ratio, min_observed=max(1, L // 10))
        miss = 1.0 - obs
        pred = model(x, obs)
        loss = masked_mse(pred, x, miss)
        total += loss.item() * B
        n += B
    return total / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="test_data/ETT-small/ETTh1.csv")
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--step", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mask_ratio", type=float, default=0.2)
    p.add_argument("--patch_len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--e_layers", type=int, default=3)
    p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=0)
    args = p.parse_args()

    data_path = Path(args.data_path)
    if not data_path.is_file():
        data_path = _ROOT / args.data_path
    if not data_path.is_file():
        raise FileNotFoundError(f"data_path not found: {args.data_path}")

    raw, col_names = load_multivariate_csv(str(data_path))
    T, n_vars = raw.shape
    train_r, val_r, test_r = time_splits(T, SplitConfig())

    train_ds = SlidingWindowDataset(raw, args.seq_len, step=args.step, start=train_r[0], end=train_r[1])
    val_ds = SlidingWindowDataset(raw, args.seq_len, step=args.step, start=val_r[0], end=val_r[1])
    test_ds = SlidingWindowDataset(raw, args.seq_len, step=args.step, start=test_r[0], end=test_r[1])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device)
    model = PatchTSTImputer(
        n_vars=n_vars,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    min_obs = max(2, args.seq_len // 20)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x in train_loader:
            x = x.to(device)
            obs = make_train_mask(x, args.mask_ratio, min_observed=min_obs)
            miss = 1.0 - obs
            pred = model(x, obs)
            loss = masked_mse(pred, x, miss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        tr = float(np.mean(losses)) if losses else 0.0
        va = evaluate(model, val_loader, device, args.mask_ratio)
        print(f"epoch {epoch:03d}  train_mse {tr:.6f}  val_mse {va:.6f}  n_vars={n_vars}  cols={len(col_names)}")

    te = evaluate(model, test_loader, device, args.mask_ratio)
    print(f"test_mse (held-out points, mask_ratio={args.mask_ratio}) {te:.6f}")


if __name__ == "__main__":
    main()
