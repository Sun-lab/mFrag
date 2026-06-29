"""mFrag-Net inference API (3-channel CNN)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from mfrag.mfrag_net.architecture import (
    FRAG_FINAL_EPOCHS,
    FRAG_FINAL_LR,
    FRAG_FINAL_WD,
    MOTIF_FINAL_EPOCHS,
    MOTIF_FINAL_LR,
    MOTIF_FINAL_WD,
    Preset29ThreeChannelCNN,
    ThreeChannelZscoreDataset,
    build_preset29_cfg,
)
from mfrag.paths import (
    LUNG_TEST_FRAG,
    LUNG_TEST_HREF,
    LUNG_TEST_LREF,
    LUNG_TEST_MOTIF,
    MFRAG_NET_FRAG_LUNG,
    MFRAG_NET_MOTIF_LUNG,
)


class _MFragNetBase:
    default_weights: Path
    default_main_dir: Path
    final_epochs: int
    final_lr: float
    final_wd: float

    def __init__(
        self,
        *,
        model_dir: Path | None = None,
        main_dir: Path | None = None,
        healthy_ref_dir: Path | None = None,
        lung_ref_dir: Path | None = None,
        device: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir or self.default_weights.parent)
        weights_pt = self.model_dir / self.default_weights.name
        self.main_dir = Path(main_dir or self.default_main_dir)
        self.healthy_ref_dir = Path(healthy_ref_dir or LUNG_TEST_HREF)
        self.lung_ref_dir = Path(lung_ref_dir or LUNG_TEST_LREF)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        cfg = replace(
            build_preset29_cfg(self.final_epochs),
            lr=float(self.final_lr),
            weight_decay=float(self.final_wd),
        )
        self.model = Preset29ThreeChannelCNN(cfg).to(self.device)
        ckpt = torch.load(weights_pt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @classmethod
    def from_pretrained(cls, **kwargs) -> "_MFragNetBase":
        return cls(**kwargs)

    def predict_proba(self, sample_stem: str) -> float:
        frag_path = self.main_dir / f"{sample_stem}.csv"
        ds = ThreeChannelZscoreDataset(
            self.main_dir,
            self.healthy_ref_dir,
            self.lung_ref_dir,
            [frag_path],
        )
        x, _ = ds[0]
        with torch.no_grad():
            logit = self.model(x.unsqueeze(0).to(self.device)).squeeze()
        return float(torch.sigmoid(logit).item())

    def predict_proba_batch(self, sample_stems: list[str]) -> np.ndarray:
        return np.asarray([self.predict_proba(s) for s in sample_stems], dtype=np.float64)


class MFragNetFragLength(_MFragNetBase):
    default_weights = MFRAG_NET_FRAG_LUNG / "rep_031_weights.pt"
    default_main_dir = LUNG_TEST_FRAG
    final_epochs = FRAG_FINAL_EPOCHS
    final_lr = FRAG_FINAL_LR
    final_wd = FRAG_FINAL_WD


class MFragNetEnd4Motif(_MFragNetBase):
    default_weights = MFRAG_NET_MOTIF_LUNG / "rep_006_weights.pt"
    default_main_dir = LUNG_TEST_MOTIF
    final_epochs = MOTIF_FINAL_EPOCHS
    final_lr = MOTIF_FINAL_LR
    final_wd = MOTIF_FINAL_WD
