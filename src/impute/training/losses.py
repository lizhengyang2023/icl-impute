from __future__ import annotations

import torch


def masked_mse(pred: torch.Tensor, target: torch.Tensor, miss: torch.Tensor) -> torch.Tensor:
    """miss: 1 on positions to score (held-out / missing)."""
    d = (pred - target).pow(2) * miss
    den = miss.sum().clamp(min=1.0)
    return d.sum() / den
