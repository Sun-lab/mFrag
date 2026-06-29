#!/usr/bin/env python3
"""
Per-sample Shannon entropy of weighted 5' end 4-mer motifs across 5 Mb windows.

- Each fragment: end_4_motif_5p + end_4_motif_5m (weighted).
- Drop windows where any training sample has NA (no fragments / zero total weight).
- Within-sample z-score across retained windows; plot lines by group (Healthy vs Lung cancer).
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_scripts = Path(__file__).resolve().parents[1] / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from meth_beta_colored_loess_mean import plot_black_loess_mean  # noqa: E402
from nmb_window_ratios_core import (  # noqa: E402
    CHROM_SIZES,
    assign_bins_vectorized,
    build_chrom_bin_offset,
    create_genomic_bins,
)

MATHIOS = Path(__file__).resolve().parents[1]
TRAIN_FRAG = MATHIOS / "fragments_csv" / "training"
OUT_DIR = Path(__file__).resolve().parent
CACHE_NPZ = OUT_DIR / "end4motif_training_window_shannon.npz"
OUT_PNG = OUT_DIR / "end4motif_training_shannon_entropy_by_window.png"
MOTIFS = ["".join(p) for p in product("ACGT", repeat=4)]
MOTIF_INDEX = {m: i for i, m in enumerate(MOTIFS)}
MOTIF_COLS = ["end_4_motif_5p", "end_4_motif_5m"]
BIN_SIZE = 5_000_000
CHUNK = 2_000_000
MIN_WINDOWS_FOR_Z = 10

BINS = create_genomic_bins(BIN_SIZE)
CHROM_OFFSET = build_chrom_bin_offset(BINS)
N_WIN = len(BINS)
WINDOW_IDS = [b[3] for b in BINS]


def _window_label(wid: str) -> str:
    parts = wid.split("_")
    if len(parts) < 3:
        return wid
    chrom_disp = parts[0][0].upper() + parts[0][1:]

    def _fmt_mb(x: float) -> str:
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.1f}".rstrip("0").rstrip(".")

    s = int(parts[1]) / 1_000_000
    e = int(parts[2]) / 1_000_000
    return f"{chrom_disp}: {_fmt_mb(s)}-{_fmt_mb(e)}Mb"


def _shannon_entropy(weights: np.ndarray) -> float:
    total = weights.sum()
    if total <= 0:
        return np.nan
    p = weights[weights > 0] / total
    return float(-np.sum(p * np.log2(p)))


def _sample_window_shannon(path_str: str) -> tuple[str, str, np.ndarray]:
    path = Path(path_str)
    acc = np.zeros((N_WIN, 256), dtype=np.float64)

    for chunk in pd.read_csv(
        path,
        usecols=["chromosome", "start", "weight", *MOTIF_COLS],
        chunksize=CHUNK,
    ):
        df = chunk.rename(columns={"chromosome": "chrom"})
        df["chrom"] = df["chrom"].astype(str)
        df["chrom"] = df["chrom"].where(
            df["chrom"].str.startswith("chr"), "chr" + df["chrom"]
        )
        df = df[df["chrom"].isin(CHROM_SIZES.keys())]
        if df.empty:
            continue

        gen_bin = assign_bins_vectorized(df, CHROM_OFFSET, BIN_SIZE)
        w = df["weight"].astype(np.float64).values

        for col in MOTIF_COLS:
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
            motifs = s[valid].values
            midx = np.array([MOTIF_INDEX.get(m, -1) for m in motifs], dtype=np.int32)
            ok = midx >= 0
            if ok.any():
                np.add.at(acc, (gb[ok], midx[ok]), wt[ok])

    H = np.full(N_WIN, np.nan, dtype=np.float64)
    for wi in range(N_WIN):
        H[wi] = _shannon_entropy(acc[wi])

    group = "healthy" if path.name.endswith("_healthy.csv") else "lung_cancer"
    return path.stem, group, H


def _zscore_within_sample(mat: np.ndarray) -> np.ndarray:
    """Row-wise z-score across windows (columns)."""
    out = np.full_like(mat, np.nan, dtype=np.float64)
    for i in range(mat.shape[0]):
        row = mat[i]
        valid = np.isfinite(row)
        if valid.sum() < MIN_WINDOWS_FOR_Z:
            continue
        m = float(np.nanmean(row))
        s = float(np.nanstd(row))
        if s > 1e-10:
            out[i] = np.where(valid, (row - m) / s, np.nan)
    return out


def _load_or_build(workers: int) -> tuple[np.ndarray, np.ndarray]:
    if CACHE_NPZ.exists():
        data = np.load(CACHE_NPZ, allow_pickle=True)
        print(f"Loaded cache {CACHE_NPZ}")
        return data["healthy_shannon"], data["cancer_shannon"]

    files = sorted(TRAIN_FRAG.glob("*_healthy.csv")) + sorted(
        TRAIN_FRAG.glob("*_lung_cancer.csv")
    )
    print(f"Computing per-window Shannon from {len(files)} samples ...")
    healthy, cancer = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_sample_window_shannon, str(p)): p for p in files}
        done = 0
        for fut in as_completed(futs):
            _, group, H = fut.result()
            (healthy if group == "healthy" else cancer).append(H)
            done += 1
            if done % 20 == 0 or done == len(files):
                print(f"  processed {done}/{len(files)} samples", flush=True)

    h_mat = np.vstack(healthy)
    c_mat = np.vstack(cancer)
    np.savez(CACHE_NPZ, healthy_shannon=h_mat, cancer_shannon=c_mat)
    print(f"Wrote cache {CACHE_NPZ}")
    return h_mat, c_mat


def _chromosome_boundaries(window_ids: list[str]) -> list[tuple[int, str]]:
    """Return (x_index, chrom_label) at first window of each chromosome."""
    marks = []
    prev_chrom = None
    for i, wid in enumerate(window_ids):
        chrom = wid.split("_")[0]
        if chrom != prev_chrom:
            marks.append((i, chrom.replace("chr", "Chr")))
            prev_chrom = chrom
    return marks


def _plot_profile(
    h_z: np.ndarray,
    c_z: np.ndarray,
    window_ids: list[str],
) -> None:
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    x = np.arange(len(window_ids))
    labels = [_window_label(w) for w in window_ids]

    all_mat = np.vstack([h_z, c_z])
    global_mean = np.nanmean(all_mat, axis=0)
    h_plot = h_z - global_mean
    c_plot = c_z - global_mean

    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True, sharey=True)

    for ax, (title, mat) in zip(
        axes,
        [("Healthy", h_plot), ("Lung cancer", c_plot)],
    ):
        n = mat.shape[0]
        for row in mat:
            ax.plot(x, row, color="grey", alpha=0.45, linewidth=0.7)
        mean_vals = np.nanmean(mat, axis=0)
        plot_black_loess_mean(ax, x, mean_vals)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.set_ylabel("Shannon entropy (z-score)", fontsize=12, fontweight="bold")
        ax.set_title(f"{title} (N={n})", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")
        for chrom_i, (xi, clab) in enumerate(_chromosome_boundaries(window_ids)):
            if chrom_i > 0:
                ax.axvline(xi - 0.5, color="#888888", linestyle=":", linewidth=0.6, alpha=0.5)

    # Sparse x ticks: first window label per chromosome
    tick_pos = [m[0] for m in _chromosome_boundaries(window_ids)]
    tick_lbl = [labels[i] for i in tick_pos]
    axes[1].set_xticks(tick_pos)
    axes[1].set_xticklabels(tick_lbl, rotation=45, ha="right", fontsize=7, fontweight="bold")
    for label in axes[1].get_xticklabels():
        label.set_fontweight("bold")

    axes[0].set_ylim(-2, 2)
    axes[1].set_ylim(-2, 2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle(
        "5' end 4-mer Shannon entropy by 5 Mb window",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    h_raw, c_raw = _load_or_build(args.workers)
    all_raw = np.vstack([h_raw, c_raw])
    keep = np.all(np.isfinite(all_raw), axis=0)
    n_drop = int((~keep).sum())
    print(f"Windows kept: {keep.sum()} / {N_WIN} (dropped {n_drop} with any sample NA)")

    if keep.sum() < MIN_WINDOWS_FOR_Z:
        raise SystemExit("Too few windows after filtering")

    win_ids = [WINDOW_IDS[i] for i in np.where(keep)[0]]
    h_f = h_raw[:, keep]
    c_f = c_raw[:, keep]

    h_z = _zscore_within_sample(h_f)
    c_z = _zscore_within_sample(c_f)
    _plot_profile(h_z, c_z, win_ids)


if __name__ == "__main__":
    main()
