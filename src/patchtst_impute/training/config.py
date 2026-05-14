from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Hyperparameters for self-supervised PatchTST imputer training."""

    data_path: str
    seq_len: int = 96
    step: int = 4
    batch_size: int = 64
    epochs: int = 5
    lr: float = 1e-3
    mask_ratio: float = 0.2
    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1
    device: str = ""
    num_workers: int = 0
    log_dir: str = "src/logs"
    log_file: str = "train.log"
    log_level: str = "INFO"
    run_name: str = ""
