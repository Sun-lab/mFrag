"""Feature bin definitions for 589 × 24 mFrag matrices."""
from __future__ import annotations

import numpy as np

N_FEATURE_BINS = 24
MIN_LEN = 100
MAX_LEN = 220
BIN_WIDTH = 5

LEN_COLS = [f"{100 + BIN_WIDTH * j}-{100 + BIN_WIDTH * (j + 1)}bp" for j in range(23)] + ["215-220bp"]

MOTIF_NAMES = [
    "CCTG", "CCCA", "CCAG", "CCAA", "CCTT", "CCAT", "CCTA", "CCTC",
    "CCCT", "CCCC", "CCAC", "TATA", "GCTG", "CTCA", "GCCT", "TATG",
    "TGTA", "TTAA", "ATAA", "TGGT", "TGTG", "AGGA", "ATAT", "TTGG",
]
MOTIF_TO_COL = {m: i for i, m in enumerate(MOTIF_NAMES)}

N_BETA_BINS = 24


def _beta_col_name(i: int) -> str:
    lo = i / N_BETA_BINS
    hi = (i + 1) / N_BETA_BINS
    if i == 0:
        return f"0-{hi:.4f}"
    if i == N_BETA_BINS - 1:
        return f"{lo:.4f}-1"
    return f"{lo:.4f}-{hi:.4f}"


BETA_COLS = [_beta_col_name(i) for i in range(N_BETA_BINS)]

SHORT_LEN_COLS = [f"{100 + 5 * j}-{100 + 5 * (j + 1)}bp" for j in range(10)]
LONG_LEN_COLS = [f"{100 + 5 * j}-{100 + 5 * (j + 1)}bp" for j in range(10, 23)] + ["215-220bp"]

MOTIF_FRAGMENT_COLS = ["end_4_motif_5p", "end_4_motif_5m"]
COL_HEALTHY_METH = "HealthyRef_DNAMethylation"
COL_LUNG_METH = "LungcancerRef_DNAMethylation"


def length_bin_index(length: np.ndarray) -> np.ndarray:
    L = length.astype(np.int64)
    valid = (L >= MIN_LEN) & (L <= MAX_LEN)
    idx = np.full(len(L), -1, dtype=np.int64)
    idx[valid] = np.minimum((L[valid] - MIN_LEN) // BIN_WIDTH, N_FEATURE_BINS - 1)
    return idx


def beta_bin_index(beta: np.ndarray) -> np.ndarray:
    b = beta.astype(np.float64)
    ok = np.isfinite(b) & (b >= 0.0) & (b <= 1.0)
    idx = np.full(len(b), -1, dtype=np.int64)
    if ok.any():
        bi = np.floor(b[ok] * N_BETA_BINS).astype(np.int64)
        idx[ok] = np.minimum(bi, N_BETA_BINS - 1)
    return idx
