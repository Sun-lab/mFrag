"""Build 589×24 per-sample matrices for mFrag-Net."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mfrag.preprocessing.feature_bins import (
    BETA_COLS,
    COL_HEALTHY_METH,
    COL_LUNG_METH,
    LEN_COLS,
    MAX_LEN,
    MIN_LEN,
    MOTIF_FRAGMENT_COLS,
    MOTIF_NAMES,
    MOTIF_TO_COL,
    beta_bin_index,
    length_bin_index,
)
from mfrag.preprocessing.windows import (
    BIN_SIZE,
    CHROM_SIZES,
    assign_bins_vectorized,
    build_chrom_bin_offset,
    create_genomic_bins,
    normalize_chrom,
)

CHUNK_ROWS = 500_000


def _write_matrix_triplet(
    acc: np.ndarray,
    window_ids: list[str],
    feature_cols: list[str],
    out_raw: Path,
    out_row: Path,
    out_z: Path,
    stem: str,
) -> None:
    for d in (out_raw, out_row, out_z):
        d.mkdir(parents=True, exist_ok=True)

    raw_df = pd.DataFrame(acc, columns=feature_cols)
    raw_df.insert(0, "window_id", window_ids)
    raw_df.to_csv(out_raw / f"{stem}.csv", index=False)

    row_sum = acc.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        prop = np.where(row_sum > 0, acc / row_sum, 0.0)
    row_df = pd.DataFrame(prop, columns=feature_cols)
    row_df.insert(0, "window_id", window_ids)
    row_df.to_csv(out_row / f"{stem}.csv", index=False)

    flat = acc.ravel()
    mu, sd = float(np.mean(flat)), float(np.std(flat))
    zmat = np.zeros_like(acc, dtype=np.float64) if (not np.isfinite(sd) or sd == 0.0) else (acc - mu) / sd
    z_df = pd.DataFrame(zmat, columns=feature_cols)
    z_df.insert(0, "window_id", window_ids)
    z_df.to_csv(out_z / f"{stem}.csv", index=False)


def build_fragment_length_matrices(
    fragments_csv: Path | str,
    out_dir: Path | str,
    *,
    write_raw: bool = False,
    write_row_norm: bool = False,
) -> Path:
    """Build fragment-length 589×24 matrices; returns path to z-scored CSV."""
    src = Path(fragments_csv)
    out = Path(out_dir)
    stem = src.stem
    bins = create_genomic_bins()
    chrom_offset = build_chrom_bin_offset(bins)
    n_win = len(bins)
    window_ids = [b[3] for b in bins]
    acc = np.zeros((n_win, len(LEN_COLS)), dtype=np.float64)

    for chunk in pd.read_csv(
        src,
        usecols=["chromosome", "start", "fragment_length", "weight"],
        chunksize=CHUNK_ROWS,
    ):
        df = chunk.rename(columns={"chromosome": "chrom"})
        df["chrom"] = normalize_chrom(df["chrom"])
        df = df[df["chrom"].isin(CHROM_SIZES)]
        fl = df["fragment_length"].values
        df = df[(fl >= MIN_LEN) & (fl <= MAX_LEN)]
        if df.empty:
            continue
        gen_bin = assign_bins_vectorized(df, chrom_offset)
        lb = length_bin_index(df["fragment_length"].values)
        w = df["weight"].to_numpy(dtype=np.float64)
        ok = (gen_bin >= 0) & (lb >= 0)
        if ok.any():
            np.add.at(acc, (gen_bin[ok], lb[ok]), w[ok])

    z_dir = out / "fragment_length_zscored"
    raw_dir = out / "fragment_length_raw_weights" if write_raw else z_dir
    row_dir = out / "fragment_length_row_norm" if write_row_norm else z_dir
    _write_matrix_triplet(acc, window_ids, LEN_COLS, raw_dir, row_dir, z_dir, stem)
    return z_dir / f"{stem}.csv"


def build_end4motif_matrices(
    fragments_csv: Path | str,
    out_dir: Path | str,
    *,
    write_raw: bool = False,
    write_row_norm: bool = False,
) -> Path:
    """Build end-4-mer motif 589×24 matrices; returns path to z-scored CSV."""
    src = Path(fragments_csv)
    out = Path(out_dir)
    stem = src.stem
    bins = create_genomic_bins()
    chrom_offset = build_chrom_bin_offset(bins)
    n_win = len(bins)
    window_ids = [b[3] for b in bins]
    acc = np.zeros((n_win, len(MOTIF_NAMES)), dtype=np.float64)

    for chunk in pd.read_csv(
        src,
        usecols=["chromosome", "start", "fragment_length", "weight", *MOTIF_FRAGMENT_COLS],
        chunksize=CHUNK_ROWS,
    ):
        df = chunk.rename(columns={"chromosome": "chrom"})
        df["chrom"] = normalize_chrom(df["chrom"])
        df = df[df["chrom"].isin(CHROM_SIZES)]
        fl = df["fragment_length"].values
        df = df[(fl >= MIN_LEN) & (fl <= MAX_LEN)]
        if df.empty:
            continue
        gen_bin = assign_bins_vectorized(df, chrom_offset)
        w = df["weight"].to_numpy(dtype=np.float64)
        ok_win = gen_bin >= 0
        for col in MOTIF_FRAGMENT_COLS:
            motifs = df[col].astype(str).str.upper().str.strip()
            valid = (
                ok_win
                & df[col].notna()
                & motifs.str.len().eq(4)
                & motifs.str.match(r"^[ACGT]{4}$", na=False)
            )
            if not valid.any():
                continue
            gb = gen_bin[valid]
            wt = w[valid]
            col_idx = np.array([MOTIF_TO_COL.get(m, -1) for m in motifs[valid].values], dtype=np.int64)
            m2 = col_idx >= 0
            if m2.any():
                np.add.at(acc, (gb[m2], col_idx[m2]), wt[m2])

    z_dir = out / "end4motif_zscored"
    raw_dir = out / "end4motif_raw_weights" if write_raw else z_dir
    row_dir = out / "end4motif_row_norm" if write_row_norm else z_dir
    _write_matrix_triplet(acc, window_ids, MOTIF_NAMES, raw_dir, row_dir, z_dir, stem)
    return z_dir / f"{stem}.csv"


def build_reference_methylation_matrices(
    fragments_csv: Path | str,
    out_dir: Path | str,
    *,
    write_raw: bool = False,
    write_row_norm: bool = False,
) -> tuple[Path, Path]:
    """Build healthy/lung reference methylation 589×24 matrices; returns z-scored paths."""
    src = Path(fragments_csv)
    out = Path(out_dir)
    stem = src.stem
    bins = create_genomic_bins()
    chrom_offset = build_chrom_bin_offset(bins)
    n_win = len(bins)
    window_ids = [b[3] for b in bins]
    acc_h = np.zeros((n_win, len(BETA_COLS)), dtype=np.float64)
    acc_l = np.zeros((n_win, len(BETA_COLS)), dtype=np.float64)

    usecols = ["chromosome", "start", "fragment_length", COL_HEALTHY_METH, COL_LUNG_METH, "weight"]
    for chunk in pd.read_csv(src, usecols=usecols, chunksize=CHUNK_ROWS, low_memory=False):
        df = chunk.rename(columns={"chromosome": "chrom"})
        df["chrom"] = normalize_chrom(df["chrom"])
        df = df[df["chrom"].isin(CHROM_SIZES)]
        fl = pd.to_numeric(df["fragment_length"], errors="coerce").to_numpy(dtype=np.float64)
        df = df[(fl >= MIN_LEN) & (fl <= MAX_LEN)]
        if df.empty:
            continue
        wt = pd.to_numeric(df["weight"], errors="coerce").to_numpy(dtype=np.float64)
        w_ok = np.isfinite(wt) & (wt > 0)
        h_beta = pd.to_numeric(df[COL_HEALTHY_METH], errors="coerce").to_numpy()
        l_beta = pd.to_numeric(df[COL_LUNG_METH], errors="coerce").to_numpy()
        gen_bin = assign_bins_vectorized(df, chrom_offset)
        base_ok = (gen_bin >= 0) & w_ok
        bh = beta_bin_index(h_beta)
        ok_h = base_ok & (bh >= 0)
        if ok_h.any():
            np.add.at(acc_h, (gen_bin[ok_h], bh[ok_h]), wt[ok_h])
        bl = beta_bin_index(l_beta)
        ok_l = base_ok & (bl >= 0)
        if ok_l.any():
            np.add.at(acc_l, (gen_bin[ok_l], bl[ok_l]), wt[ok_l])

    h_z = out / "healthy_ref_methylation_zscored"
    l_z = out / "lung_ref_methylation_zscored"
    h_raw = out / "healthy_ref_methylation_raw_weights" if write_raw else h_z
    h_row = out / "healthy_ref_methylation_row_norm" if write_row_norm else h_z
    l_raw = out / "lung_ref_methylation_raw_weights" if write_raw else l_z
    l_row = out / "lung_ref_methylation_row_norm" if write_row_norm else l_z
    _write_matrix_triplet(acc_h, window_ids, BETA_COLS, h_raw, h_row, h_z, stem)
    _write_matrix_triplet(acc_l, window_ids, BETA_COLS, l_raw, l_row, l_z, stem)
    return h_z / f"{stem}.csv", l_z / f"{stem}.csv"
