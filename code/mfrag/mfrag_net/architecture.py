"""Preset-29 three-channel CNN architecture for mFrag-Net inference."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

N_WINDOWS = 589
N_LEN_BINS = 24

S1_ARCH19_CHANNELS = (72, 72, 56, 40, 24)
S1_ARCH19_KERNELS = (6, 6, 6, 6, 4)
LR_P29 = 0.017930324031
FIXED_WD = 9.9e-7

# Trained lung-cancer model hyperparameters (frag-length and end4motif variants)
FRAG_FINAL_EPOCHS = 90
FRAG_FINAL_LR = 0.01455
FRAG_FINAL_WD = 3e-6
MOTIF_FINAL_EPOCHS = 90
MOTIF_FINAL_LR = 0.01455
MOTIF_FINAL_WD = 3e-6


def label_from_stem(stem: str) -> float:
    if "lung_cancer" in stem:
        return 1.0
    if "_healthy" in stem or stem.endswith("healthy"):
        return 0.0
    raise ValueError(f"Cannot infer label from sample stem: {stem}")


def conv1d_out_len(l_in: int, kernel: int, stride: int, padding: int = 0, dilation: int = 1) -> int:
    return (l_in + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def stage2_ok(width: int, n_layers: int, kernel: int, stride: int) -> bool:
    L = width
    for _ in range(n_layers):
        if L < kernel:
            return False
        L = conv1d_out_len(L, kernel, stride, 0, 1)
        if L < 1:
            return False
    return True


@dataclass
class SweepConfig:
    s1_channels: tuple[int, ...]
    s1_kernel: int
    s1_activation: str
    s2_layers: int
    s2_channels: int
    s2_kernel: int
    s2_stride: int
    dropout_s1: float
    dropout_emb: float
    head_hidden: int
    dropout_head: float
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    grad_clip: float
    use_scheduler: bool
    residual_s1: bool = False
    residual_s2: bool = False
    s1_kernels: Optional[tuple[int, ...]] = None
    s1_dropout_per_block: Optional[tuple[float, ...]] = None
    s2_out_channels: Optional[tuple[int, ...]] = None
    head_hidden2: int = 0


def build_preset29_cfg(epochs: int) -> SweepConfig:
    return SweepConfig(
        s1_channels=S1_ARCH19_CHANNELS,
        s1_kernel=6,
        s1_kernels=S1_ARCH19_KERNELS,
        s1_activation="gelu",
        s2_layers=5,
        s2_channels=56,
        s2_kernel=7,
        s2_stride=2,
        dropout_s1=0.15,
        dropout_emb=0.18,
        head_hidden=96,
        dropout_head=0.35,
        lr=float(LR_P29),
        weight_decay=FIXED_WD,
        batch_size=12,
        epochs=int(epochs),
        grad_clip=2.5,
        use_scheduler=True,
    )


def sweep_active_n_windows(cfg: SweepConfig) -> int:
    return N_WINDOWS


class _Stage1Block(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int, act: nn.Module, dropout: float, use_residual: bool):
        super().__init__()
        self.use_residual = use_residual
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(c_out)
        if use_residual:
            self.skip = nn.Conv1d(c_in, c_out, kernel_size=k, stride=1, padding=0)
            self.bn_skip = nn.BatchNorm1d(c_out)
        self.act = act
        self.drop = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bn(self.conv(x))
        if self.use_residual:
            h = h + self.bn_skip(self.skip(x))
        h = self.act(h)
        return self.drop(h)


class Stage1Encoder(nn.Module):
    def __init__(
        self,
        channels: tuple[int, ...],
        kernel_size: int,
        activation: str = "relu",
        dropout: float = 0.0,
        use_residual: bool = False,
        kernel_sizes: Optional[tuple[int, ...]] = None,
        dropout_per_block: Optional[tuple[float, ...]] = None,
        in_channels: int = 1,
    ):
        super().__init__()
        acts = {"relu": nn.ReLU(inplace=True), "gelu": nn.GELU()}
        act = acts[activation]
        ks = kernel_sizes if kernel_sizes is not None else (kernel_size,) * len(channels)
        drops = dropout_per_block if dropout_per_block is not None else (dropout,) * len(channels)
        blocks: list[nn.Module] = []
        c_in = in_channels
        for c_out, k, drop_p in zip(channels, ks, drops):
            blocks.append(_Stage1Block(c_in, c_out, k, act, drop_p, use_residual))
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)
        self._out_ch = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x.mean(dim=2)


class _Stage2Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int, use_residual: bool):
        super().__init__()
        self.use_residual = use_residual
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=0)
        self.bn = nn.BatchNorm1d(out_ch)
        if use_residual:
            self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=0)
            self.bn_skip = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bn(self.conv(x))
        if self.use_residual:
            h = h + self.bn_skip(self.skip(x))
        return self.act(h)


class Preset29ThreeChannelCNN(nn.Module):
    """(B, W, 24, 3) fragment + dual-reference methylation → cancer logit."""

    def __init__(self, cfg: SweepConfig):
        super().__init__()
        self.cfg = cfg
        self._n_win = sweep_active_n_windows(cfg)
        self.stage1 = Stage1Encoder(
            cfg.s1_channels,
            cfg.s1_kernel,
            activation=cfg.s1_activation,
            dropout=cfg.dropout_s1,
            use_residual=cfg.residual_s1,
            kernel_sizes=cfg.s1_kernels,
            dropout_per_block=cfg.s1_dropout_per_block,
            in_channels=3,
        )
        s1_out = self.stage1._out_ch
        s2_outs = cfg.s2_out_channels if cfg.s2_out_channels is not None else (cfg.s2_channels,) * cfg.s2_layers
        stages2: list[nn.Module] = []
        in_ch = s1_out
        for out_ch in s2_outs:
            stages2.append(_Stage2Block(in_ch, out_ch, cfg.s2_kernel, cfg.s2_stride, cfg.residual_s2))
            in_ch = out_ch
        self.stage2 = nn.Sequential(*stages2)
        s2_final_d = s2_outs[-1]
        self.drop_emb = nn.Dropout(cfg.dropout_emb) if cfg.dropout_emb > 0 else nn.Identity()
        d_head = cfg.dropout_head
        self.head = nn.Sequential(
            nn.Dropout(d_head) if d_head > 0 else nn.Identity(),
            nn.Linear(s2_final_d, cfg.head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(d_head) if d_head > 0 else nn.Identity(),
            nn.Linear(cfg.head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, w, l, c = x.shape
        x_in = x.permute(0, 1, 3, 2).reshape(b * w, 3, l)
        emb = self.stage1(x_in)
        emb = self.drop_emb(emb)
        emb = emb.view(b, self.stage1._out_ch, w)
        x2 = self.stage2(emb)
        x2 = x2.mean(dim=2)
        return self.head(x2).squeeze(-1)


class ThreeChannelZscoreDataset(Dataset):
    """Stack fragment-length, healthy-ref, and lung-ref z-scored 589×24 matrices."""

    def __init__(
        self,
        frag_dir: Path,
        healthy_beta_dir: Path,
        lung_beta_dir: Path,
        paths: list[Path],
    ):
        self.paths = sorted(paths)
        self.frag_dir = frag_dir
        self.healthy_beta_dir = healthy_beta_dir
        self.lung_beta_dir = lung_beta_dir

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.paths[idx]
        stem = p.stem
        d0 = pd.read_csv(self.frag_dir / f"{stem}.csv")
        d1 = pd.read_csv(self.healthy_beta_dir / f"{stem}.csv")
        d2 = pd.read_csv(self.lung_beta_dir / f"{stem}.csv")
        a0 = d0.iloc[:, 1:].to_numpy(dtype=np.float32)
        a1 = d1.iloc[:, 1:].to_numpy(dtype=np.float32)
        a2 = d2.iloc[:, 1:].to_numpy(dtype=np.float32)
        x = np.stack([a0, a1, a2], axis=-1)
        y = label_from_stem(stem)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)
