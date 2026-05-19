from __future__ import annotations

import torch


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
