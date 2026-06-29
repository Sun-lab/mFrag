"""Per-window methylation coupling weights for mFrag-Weight (lung cancer)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

FACTOR_FLOOR = 0.05
MDS_LAMBDA = 0.5


def load_external_methylation(path: Path) -> pd.DataFrame:
    meth = pd.read_csv(path)
    meth["abs_delta_beta"] = (
        meth["healthy_mean_beta"] - meth["lung_cancer_mean_beta"]
    ).abs()
    return meth


def sl_ratio_weights(window_ids: list[str], meth: pd.DataFrame) -> tuple[np.ndarray, float]:
    """S/L coupling: |Δβ| / mean(|Δβ|); windows without coverage get weight 1."""
    abs_db = meth.set_index("window_id")["abs_delta_beta"].reindex(window_ids).to_numpy(dtype=np.float64)
    finite = np.isfinite(abs_db)
    mean_abs = float(np.nanmean(abs_db[finite])) if finite.any() else 1.0
    w = np.ones_like(abs_db)
    if finite.any() and mean_abs > 0:
        w[finite] = abs_db[finite] / mean_abs
    return w, mean_abs


def mds_db_tilde(window_ids: list[str], meth: pd.DataFrame) -> np.ndarray:
    abs_db = meth.set_index("window_id")["abs_delta_beta"].reindex(window_ids).to_numpy(dtype=np.float64)
    finite = np.isfinite(abs_db)
    db_tilde = np.zeros_like(abs_db)
    if finite.any():
        r = rankdata(abs_db[finite], method="average")
        lo, hi = r.min(), r.max()
        if hi > lo:
            db_tilde[finite] = ((r - lo) / (hi - lo)).astype(np.float64)
    return db_tilde


def mds_weights(window_ids: list[str], meth: pd.DataFrame, *, lam: float = MDS_LAMBDA) -> np.ndarray:
    """MDS coupling: max(1 − λ × Δβ̃, 0.05)."""
    db_tilde = mds_db_tilde(window_ids, meth)
    return np.maximum(1.0 - lam * db_tilde, FACTOR_FLOOR)


def build_supplementary_file_1(
  meth_csv: Path,
  out_csv: Path,
  *,
  window_order: list[str] | None = None,
  mds_lambda: float = MDS_LAMBDA,
) -> pd.DataFrame:
    meth = load_external_methylation(meth_csv)
    if window_order is None:
        window_order = meth["window_id"].astype(str).tolist()
    sl_w, mean_abs = sl_ratio_weights(window_order, meth)
    mds_w = mds_weights(window_order, meth, lam=mds_lambda)
    db_tilde = mds_db_tilde(window_order, meth)
    abs_db = meth.set_index("window_id")["abs_delta_beta"].reindex(window_order)

    df = pd.DataFrame(
        {
            "window_id": window_order,
            "healthy_mean_beta": meth.set_index("window_id")["healthy_mean_beta"].reindex(window_order).values,
            "lung_cancer_mean_beta": meth.set_index("window_id")["lung_cancer_mean_beta"].reindex(window_order).values,
            "abs_delta_beta": abs_db.values,
            "sl_ratio_coupling_weight": sl_w,
            "mds_db_tilde": db_tilde,
            "mds_coupling_weight": mds_w,
        }
    )
    df.attrs["mean_abs_delta_beta"] = mean_abs
    df.attrs["mds_lambda"] = mds_lambda
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
