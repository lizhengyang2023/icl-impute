"""
Train PatchTST-style imputer with random point masking (self-supervised on fully observed CSV).
Run from repo root:
  python -m impute.train --data_path test_data/ETT-small/ETTh1.csv
(add src to PYTHONPATH, or run from src directory)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Allow `python src/impute/train.py` and `python -m impute.train`
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from impute.training.config import TrainConfig
from impute.training.engine import run_training


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
    p.add_argument("--log_dir", type=str, default="src/logs", help="Directory for log file (relative to repo root or absolute).")
    p.add_argument("--log_file", type=str, default="train.log", help="Log filename under log_dir; opened in append mode.")
    p.add_argument("--log_level", type=str, default="INFO", help="Logging level, e.g. DEBUG, INFO, WARNING.")
    p.add_argument("--run_name", type=str, default="", help="Optional label logged at run start for grep-friendly runs.")
    args = p.parse_args()

    cfg = TrainConfig(
        data_path=args.data_path,
        seq_len=args.seq_len,
        step=args.step,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        mask_ratio=args.mask_ratio,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        device=args.device,
        num_workers=args.num_workers,
        log_dir=args.log_dir,
        log_file=args.log_file,
        log_level=args.log_level,
        run_name=args.run_name,
    )
    run_training(_ROOT, cfg)


if __name__ == "__main__":
    main()
