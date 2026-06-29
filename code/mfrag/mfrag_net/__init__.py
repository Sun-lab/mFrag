"""mFrag-Net inference API (3-channel CNN)."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from mfrag.paths import (
    LUNG_TEST_FRAG,
    LUNG_TEST_HREF,
    LUNG_TEST_LREF,
    LUNG_TEST_MOTIF,
    MATHIOS_SCRIPTS,
    MFRAG_NET_FRAG_LUNG,
    MFRAG_NET_MOTIF_LUNG,
)

if str(MATHIOS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MATHIOS_SCRIPTS))

from cnn_sweep_preset29_frag3ch_dual_beta_z import (  # noqa: E402
    Preset29ThreeChannelCNN,
    ThreeChannelZscoreDataset,
    build_preset29_cfg,
)
from fdconv_final24_frag_meth3ch_new import (  # noqa: E402
    FINAL_EPOCHS as FRAG_EPOCHS,
    FINAL_LR as FRAG_LR,
    FINAL_WD as FRAG_WD,
)
from fdconv_final24_motif_meth3ch_new import (  # noqa: E402
    FINAL_EPOCHS as MOTIF_EPOCHS,
    FINAL_LR as MOTIF_LR,
    FINAL_WD as MOTIF_WD,
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
    final_epochs = FRAG_EPOCHS
    final_lr = FRAG_LR
    final_wd = FRAG_WD


class MFragNetEnd4Motif(_MFragNetBase):
    default_weights = MFRAG_NET_MOTIF_LUNG / "rep_006_weights.pt"
    default_main_dir = LUNG_TEST_MOTIF
    final_epochs = MOTIF_EPOCHS
    final_lr = MOTIF_LR
    final_wd = MOTIF_WD
