from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from patchtst_impute.log_utils import setup_logging
from patchtst_impute.models import PatchTSTImputer
from patchtst_impute.training.config import TrainConfig
from patchtst_impute.training.dataloaders import build_time_series_loaders
from patchtst_impute.training.losses import masked_mse
from patchtst_impute.training.masking import make_train_mask

logger = logging.getLogger(__name__)


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


def run_training(repo_root: Path, cfg: TrainConfig) -> None:
    """End-to-end self-supervised training on a fully observed CSV series."""
    log_dir = Path(cfg.log_dir)
    if not log_dir.is_absolute():
        log_dir = repo_root / log_dir
    setup_logging(log_dir, cfg.log_level, cfg.log_file)

    logger.info("=" * 60)
    logger.info(
        "run_start run_name=%r data_path=%s log_path=%s",
        cfg.run_name or None,
        cfg.data_path,
        str(log_dir / cfg.log_file),
    )
    device_str = cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "torch=%s cuda_available=%s device_arg=%s resolved=%s",
        torch.__version__,
        torch.cuda.is_available(),
        cfg.device or None,
        device_str,
    )
    logger.info(
        "hparams seq_len=%s step=%s batch_size=%s epochs=%s lr=%s mask_ratio=%s "
        "patch_len=%s stride=%s d_model=%s n_heads=%s e_layers=%s d_ff=%s dropout=%s num_workers=%s",
        cfg.seq_len,
        cfg.step,
        cfg.batch_size,
        cfg.epochs,
        cfg.lr,
        cfg.mask_ratio,
        cfg.patch_len,
        cfg.stride,
        cfg.d_model,
        cfg.n_heads,
        cfg.e_layers,
        cfg.d_ff,
        cfg.dropout,
        cfg.num_workers,
    )

    train_loader, val_loader, test_loader, n_vars, col_names = build_time_series_loaders(
        cfg.data_path,
        repo_root,
        seq_len=cfg.seq_len,
        step=cfg.step,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    device = torch.device(device_str)
    model = PatchTSTImputer(
        n_vars=n_vars,
        patch_len=cfg.patch_len,
        stride=cfg.stride,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        e_layers=cfg.e_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    min_obs = max(2, cfg.seq_len // 20)
    t0 = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for x in train_loader:
            x = x.to(device)
            obs = make_train_mask(x, cfg.mask_ratio, min_observed=min_obs)
            miss = 1.0 - obs
            pred = model(x, obs)
            loss = masked_mse(pred, x, miss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        tr = float(np.mean(losses)) if losses else 0.0
        va = evaluate(model, val_loader, device, cfg.mask_ratio)
        elapsed = time.perf_counter() - t0
        logger.info(
            "epoch=%03d train_mse=%.6f val_mse=%.6f n_vars=%s cols=%s elapsed_s=%.1f",
            epoch,
            tr,
            va,
            n_vars,
            len(col_names),
            elapsed,
        )

    te = evaluate(model, test_loader, device, cfg.mask_ratio)
    logger.info("test_mse=%.6f mask_ratio=%s (held-out points)", te, cfg.mask_ratio)
