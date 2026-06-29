"""mFrag-Weight inference API (PCA + Lasso on coupled window features)."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import json
import joblib
import numpy as np
import pandas as pd

from mfrag.coupling_weights import mds_weights, sl_ratio_weights
from mfrag.paths import EXT_METH_LUNG, MFRAG_WEIGHT_MDS_LUNG, MFRAG_WEIGHT_SL_LUNG


def _apply_coupling(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return X * weights[np.newaxis, :]


class _MFragWeightBase:
    model_subdir: Path
    coupling_kind: str  # "sl" or "mds"

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = Path(model_dir or self.model_subdir)
        self.pipeline = joblib.load(self.model_dir / "fitted_pipeline.joblib")
        meta_path = self.model_dir / "coupling_weights.json"
        self.window_ids: list[str] = []
        self.coupling_weights: np.ndarray | None = None
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            self.window_ids = list(meta["window_id"])
            self.coupling_weights = np.asarray(meta["weight"], dtype=np.float64)

    @classmethod
    def from_pretrained(cls, model_dir: Path | None = None) -> "_MFragWeightBase":
        return cls(model_dir=model_dir)

    def _couple_matrix(self, X: np.ndarray, window_ids: Sequence[str]) -> np.ndarray:
        if self.coupling_weights is not None and list(window_ids) == self.window_ids:
            w = self.coupling_weights
        else:
            meth = pd.read_csv(EXT_METH_LUNG)
            meth["abs_delta_beta"] = (meth["healthy_mean_beta"] - meth["lung_cancer_mean_beta"]).abs()
            if self.coupling_kind == "sl":
                w, _ = sl_ratio_weights(list(window_ids), meth)
            else:
                w = mds_weights(list(window_ids), meth)
        return _apply_coupling(X, w)

    def predict_proba_from_matrix(self, X: np.ndarray, window_ids: Sequence[str]) -> np.ndarray:
        Xc = self._couple_matrix(np.asarray(X, dtype=np.float64), window_ids)
        return self.pipeline.predict_proba(Xc)[:, 1]

    def predict_proba_from_wide_csv(self, wide_csv: Path) -> tuple[np.ndarray, list[str]]:
        df = pd.read_csv(wide_csv)
        names = df["sample_id"].astype(str).tolist()
        win_cols = [c for c in df.columns if c != "sample_id"]
        X = df[win_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        return self.predict_proba_from_matrix(X, win_cols), names

    def predict_proba(self, sample_row_zscore: pd.Series) -> float:
        """Predict from one sample: index = window_id, values = row-zscored feature."""
        window_ids = [str(w) for w in sample_row_zscore.index]
        X = sample_row_zscore.astype(float).to_numpy(dtype=np.float64)[np.newaxis, :]
        return float(self.predict_proba_from_matrix(X, window_ids)[0])


class MFragWeightSL(_MFragWeightBase):
    """mFrag-Weight (S/L ratio) for lung cancer."""

    model_subdir = MFRAG_WEIGHT_SL_LUNG
    coupling_kind = "sl"


class MFragWeightMDS(_MFragWeightBase):
    """mFrag-Weight (MDS / Shannon entropy) for lung cancer."""

    model_subdir = MFRAG_WEIGHT_MDS_LUNG
    coupling_kind = "mds"
