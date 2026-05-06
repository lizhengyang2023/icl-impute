"""
PatchTST-style time series imputation: patch tokens + channel independence.
Design references: Nie et al., ICLR 2023 (patching, channel-independence).
Repo: https://github.com/PatchTST/PatchTST
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean_std(x: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-(batch, channel) mean/std using only observed timesteps. x, mask: [B, L, C]."""
    m = mask.float()
    num = (x * m).sum(dim=dim, keepdim=True)
    den = m.sum(dim=dim, keepdim=True).clamp(min=1.0)
    mean = num / den
    var = ((x - mean).pow(2) * m).sum(dim=dim, keepdim=True) / den.clamp(min=1.0)
    std = (var + eps).sqrt().clamp(min=eps)
    return mean, std


def pad_to_patches(x: torch.Tensor, patch_len: int, stride: int) -> Tuple[torch.Tensor, int]:
    """Pad sequence length (dim 1) so unfold produces full patches. x: [B, L, *]."""
    L = x.size(1)
    if L < patch_len:
        pad = patch_len - L
        return F.pad(x, (0, 0, 0, pad)), pad
    rem = (L - patch_len) % stride
    if rem == 0:
        return x, 0
    pad = stride - rem
    return F.pad(x, (0, 0, 0, pad)), pad


def fold_patches(patches: torch.Tensor, patch_len: int, stride: int, length: int) -> torch.Tensor:
    """
    patches: [B, n_patches, patch_len]
    Reconstruct sequence of size `length` (may be padded target); overlap averaged.
    """
    B, n_p, pl = patches.shape
    device, dtype = patches.device, patches.dtype
    out_len = (n_p - 1) * stride + pl
    out = torch.zeros(B, out_len, device=device, dtype=dtype)
    counts = torch.zeros(B, out_len, device=device, dtype=dtype)
    for i in range(n_p):
        s = i * stride
        out[:, s : s + pl] += patches[:, i]
        counts[:, s : s + pl] += 1.0
    out = out / counts.clamp(min=1.0)
    return out[:, :length]


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len: int, d_model: int, in_dim: int = 2):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len * in_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [BC, n_patches, patch_len, in_dim]
        BC, n, pl, d = x.shape
        return self.proj(x.reshape(BC, n, pl * d))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        return x + self.pe[:, : x.size(1)]


class PatchTSTImputer(nn.Module):
    """
    Multivariate imputation with channel-independent patch Transformer.
    Input: x [B,L,C] (raw scale), mask [B,L,C] (1 = observed for conditioning).
    Output: x_hat [B,L,C] reconstructed in raw scale (same length as input; right-pad trimmed).
    """

    def __init__(
        self,
        n_vars: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        self.patch_embed = PatchEmbedding(patch_len, d_model, in_dim=2)
        self.pos = PositionalEncoding(d_model, max_len=1024)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=e_layers)
        self.head = nn.Linear(d_model, patch_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C] ground-truth scale (missing may be arbitrary; masked positions ignored for stats).
        mask: [B, L, C] float/bool, 1 = observed conditioning token.
        """
        B, L, C = x.shape
        assert C == self.n_vars

        x_obs = torch.where(mask.bool(), x, torch.zeros_like(x))
        mean, std = masked_mean_std(x, mask, dim=1)
        x_n = (x_obs - mean) / std

        # (normalized value, observed flag) per variate; channel independence (PatchTST-style).
        m_exp = mask.float()
        z = torch.stack([x_n, m_exp], dim=-1)  # [B, L, C, 2]

        z = z.permute(0, 2, 1, 3).reshape(B * C, L, 2)  # [BC, L, 2]

        z_pad, pad = pad_to_patches(z, self.patch_len, self.stride)
        Lp = z_pad.size(1)
        z_unf = z_pad.unfold(1, self.patch_len, self.stride)  # [BC, n_patches, patch_len, 2]
        n_patches = z_unf.size(1)

        tok = self.patch_embed(z_unf)  # [BC, n_p, d]
        tok = self.dropout(self.pos(tok))
        h = self.encoder(tok)
        patch_pred = self.head(h)  # [BC, n_p, patch_len] normalized space

        seq_flat = fold_patches(patch_pred, self.patch_len, self.stride, Lp)
        seq = seq_flat[:, :L]  # [BC, L]

        seq = seq.view(B, C, L).permute(0, 2, 1).contiguous()
        x_hat_n = seq
        x_hat = x_hat_n * std + mean
        return x_hat
