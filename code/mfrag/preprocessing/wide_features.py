"""Wide per-sample tables for mFrag-Weight (S/L ratio and MDS Shannon entropy)."""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from mfrag.preprocessing.feature_bins import (
    LONG_LEN_COLS,
    MOTIF_FRAGMENT_COLS,
    SHORT_LEN_COLS,
)
from mfrag.preprocessing.windows import (
    BIN_SIZE,
    CHROM_SIZES,
    assign_bins_vectorized,
    build_chrom_bin_offset,
    create_genomic_bins,
    normalize_chrom,
)

CHUNK = 2_000_000
ALL_MOTIFS = ["".join(p) for p in product("ACGT", repeat=4)]
MOTIF_INDEX = {m: i for i, m in enumerate(ALL_MOTIFS)}


def row_zscore_wide(df: pd.DataFrame, sample_col: str = "sample_id") -> pd.DataFrame:
    """Per-sample z-score across window columns (mFrag-Weight input format)."""
    out = df.copy()
    win_cols = [c for c in df.columns if c != sample_col]
    X = df[win_cols].to_numpy(dtype=np.float64)
    for i in range(len(X)):
        row = X[i]
        mu = np.nanmean(row)
        sd = np.nanstd(row, ddof=0)
        if not np.isfinite(sd) or sd == 0.0:
            X[i] = np.where(np.isfinite(row), 0.0, np.nan)
        else:
            X[i] = (row - mu) / sd
    out[win_cols] = X
    return out


def _sl_ratio_from_matrix_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = df[SHORT_LEN_COLS].to_numpy(dtype=np.float64).sum(axis=1)
    ell = df[LONG_LEN_COLS].to_numpy(dtype=np.float64).sum(axis=1)
    r = np.divide(s, ell, out=np.full_like(s, np.nan), where=ell > 0)
    return s, ell, r


def build_sl_ratio_wide_table(
    length_raw_csvs: list[Path],
    *,
    reference_training_csvs: list[Path] | None = None,
) -> pd.DataFrame:
    """
    Build wide S/L ratio table from per-sample fragment-length raw-weight matrices.

    ``reference_training_csvs`` defines the window intersection (all training samples
  with long_sum > 0). If None, intersection is computed from ``length_raw_csvs``.
    """
    files = sorted(length_raw_csvs)
    if not files:
        raise ValueError("No length raw CSV files provided")
    ref_files = sorted(reference_training_csvs) if reference_training_csvs else files

    ref = pd.read_csv(ref_files[0])
    win_ref = ref["window_id"].astype(str).tolist()
    n_win = len(win_ref)

    all_long_ok = np.ones(n_win, dtype=bool)
    for p in ref_files:
        df = pd.read_csv(p)
        if df["window_id"].astype(str).tolist() != win_ref:
            raise ValueError(f"window_id mismatch in {p.name}")
        _, long_sum, _ = _sl_ratio_from_matrix_df(df)
        all_long_ok &= long_sum > 0

    common_windows = [w for w, ok in zip(win_ref, all_long_ok) if ok]
    idx_keep = np.array([i for i, ok in enumerate(all_long_ok) if ok], dtype=int)

    rows = []
    for p in files:
        df = pd.read_csv(p)
        _, long_sum, ratio = _sl_ratio_from_matrix_df(df)
        row = {"sample_id": p.stem}
        for j, w in zip(idx_keep, common_windows):
            row[w] = ratio[j] if long_sum[j] > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)[["sample_id"] + common_windows]


def _shannon_entropy(weights: np.ndarray) -> float:
    total = weights.sum()
    if total <= 0:
        return np.nan
    p = weights[weights > 0] / total
    return float(-np.sum(p * np.log2(p)))


def sample_window_shannon_entropy(fragments_csv: Path) -> np.ndarray:
    """Per-window Shannon entropy over all 256 end 4-mers (5p + 5m weighted)."""
    bins = create_genomic_bins()
    chrom_offset = build_chrom_bin_offset(bins)
    n_win = len(bins)
    acc = np.zeros((n_win, 256), dtype=np.float64)

    for chunk in pd.read_csv(
        fragments_csv,
        usecols=["chromosome", "start", "weight", *MOTIF_FRAGMENT_COLS],
        chunksize=CHUNK,
    ):
        df = chunk.rename(columns={"chromosome": "chrom"})
        df["chrom"] = normalize_chrom(df["chrom"])
        df = df[df["chrom"].isin(CHROM_SIZES)]
        if df.empty:
            continue
        gen_bin = assign_bins_vectorized(df, chrom_offset)
        w = df["weight"].astype(np.float64).values
        for col in MOTIF_FRAGMENT_COLS:
            s = df[col].astype(str).str.upper().str.strip()
            valid = (
                df[col].notna()
                & s.str.len().eq(4)
                & s.str.match(r"^[ACGT]{4}$", na=False)
                & (gen_bin >= 0)
            )
            if not valid.any():
                continue
            gb = gen_bin[valid]
            wt = w[valid]
            midx = np.array([MOTIF_INDEX.get(m, -1) for m in s[valid].values], dtype=np.int32)
            ok = midx >= 0
            if ok.any():
                np.add.at(acc, (gb[ok], midx[ok]), wt[ok])

    H = np.full(n_win, np.nan, dtype=np.float64)
    for wi in range(n_win):
        H[wi] = _shannon_entropy(acc[wi])
    return H


def build_mds_shannon_wide_table(
    fragment_csvs: list[Path],
    *,
    reference_training_csvs: list[Path] | None = None,
) -> pd.DataFrame:
    """Build wide MDS (Shannon entropy) table from annotated fragment CSVs."""
    files = sorted(fragment_csvs)
    ref_files = sorted(reference_training_csvs) if reference_training_csvs else files
    if not files:
        raise ValueError("No fragment CSV files provided")

    bins = create_genomic_bins()
    window_ids = [b[3] for b in bins]
    n_win = len(window_ids)

    ref_ent = {p.stem: sample_window_shannon_entropy(p) for p in ref_files}
    all_finite = np.ones(n_win, dtype=bool)
    for H in ref_ent.values():
        all_finite &= np.isfinite(H) & (H > 0)

    common_windows = [w for w, ok in zip(window_ids, all_finite) if ok]
    idx_keep = np.array([i for i, ok in enumerate(all_finite) if ok], dtype=int)

    rows = []
    for p in files:
        H = sample_window_shannon_entropy(p) if p.stem not in ref_ent else ref_ent[p.stem]
        row = {"sample_id": p.stem}
        for j, w in zip(idx_keep, common_windows):
            row[w] = H[j]
        rows.append(row)
    return pd.DataFrame(rows)[["sample_id"] + common_windows]


def build_mfrag_weight_sl_ratio_csv(
    length_raw_csvs: list[Path | str],
    out_csv: Path | str,
    *,
    reference_training_csvs: list[Path | str] | None = None,
    row_zscore: bool = True,
) -> Path:
    """Write mFrag-Weight S/L ratio input CSV (row-zscored by default)."""
    raw_paths = [Path(p) for p in length_raw_csvs]
    ref_paths = [Path(p) for p in reference_training_csvs] if reference_training_csvs else None
    wide = build_sl_ratio_wide_table(raw_paths, reference_training_csvs=ref_paths)
    if row_zscore:
        wide = row_zscore_wide(wide)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out, index=False)
    return out


def build_mfrag_weight_mds_csv(
    fragment_csvs: list[Path | str],
    out_csv: Path | str,
    *,
    reference_training_csvs: list[Path | str] | None = None,
    row_zscore: bool = True,
) -> Path:
    """Write mFrag-Weight MDS (Shannon) input CSV (row-zscored by default)."""
    frag_paths = [Path(p) for p in fragment_csvs]
    ref_paths = [Path(p) for p in reference_training_csvs] if reference_training_csvs else None
    wide = build_mds_shannon_wide_table(frag_paths, reference_training_csvs=ref_paths)
    if row_zscore:
        wide = row_zscore_wide(wide)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out, index=False)
    return out
