"""Combine mFrag-Weight and mFrag-Net probabilities into the primary mFrag score (lung cancer)."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def average_proba(
    weight_proba: float | Sequence[float] | np.ndarray,
    net_proba: float | Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """
    Primary mFrag score used in the manuscript.

    Equal-weight average of mFrag-Weight and mFrag-Net predicted probabilities
    for the matched modality (fragment-length: Weight S/L + Net frag length;
    motif: Weight MDS + Net end 4-mer).

    Parameters
    ----------
    weight_proba, net_proba
        Scalar or array-like cancer probabilities in [0, 1], same shape.

    Returns
    -------
    float or ndarray
        ``0.5 * (weight_proba + net_proba)``.
    """
    w = np.asarray(weight_proba, dtype=np.float64)
    n = np.asarray(net_proba, dtype=np.float64)
    if w.shape != n.shape:
        raise ValueError(f"shape mismatch: weight {w.shape} vs net {n.shape}")
    out = 0.5 * (w + n)
    if out.ndim == 0:
        return float(out)
    return out
