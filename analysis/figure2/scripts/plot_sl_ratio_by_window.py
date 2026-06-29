#!/usr/bin/env python3
"""
GC-corrected (precomputed weight) short/long ratio by 5 Mb genomic window.

- short: fragment_length < 150; long: fragment_length > 150 (150 excluded).
- s/l = sum(short weights) / sum(long weights) per window.
- Within-sample z-score across windows; plot Healthy vs Lung cancer (grey lines).

Two profiles:
  1) fragments 100–220 bp only
  2) all fragment lengths
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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

BIN_SIZE = 5_000_000
CHUNK = 2_000_000
THRESHOLD = 150
MIN_WEIGHTED_COUNT = 50
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


def _chromosome_boundaries(window_ids: list[str]) -> list[tuple[int, str]]:
    marks = []
    prev_chrom = None
    for i, wid in enumerate(window_ids):
        chrom = wid.split("_")[0]
        if chrom != prev_chrom:
            marks.append((i, chrom.replace("chr", "Chr")))
            prev_chrom = chrom
    return marks


def _sample_window_sl(
    path_str: str, length_mode: str, unweighted: bool = False
) -> tuple[str, str, np.ndarray]:
    path = Path(path_str)
    short_acc = np.zeros(N_WIN, dtype=np.float64)
    long_acc = np.zeros(N_WIN, dtype=np.float64)

    usecols = ["chromosome", "start", "fragment_length"]
    if not unweighted:
        usecols.append("weight")
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK):
        df = chunk.rename(columns={"chromosome": "chrom"})
        df["chrom"] = df["chrom"].astype(str)
        df["chrom"] = df["chrom"].where(
            df["chrom"].str.startswith("chr"), "chr" + df["chrom"]
        )
        df = df[df["chrom"].isin(CHROM_SIZES.keys())]
        if df.empty:
            continue

        fl = df["fragment_length"].astype(np.int32)
        if length_mode == "100_220":
            in_range = (fl >= 100) & (fl <= 220)
            df = df.loc[in_range].copy()
            fl = fl[in_range]
            if df.empty:
                continue

        w = (
            np.ones(len(df), dtype=np.float64)
            if unweighted
            else df["weight"].astype(np.float64).values
        )
        gen_bin = assign_bins_vectorized(df, CHROM_OFFSET, BIN_SIZE)
        valid_bin = gen_bin >= 0

        short_mask = (fl < THRESHOLD).values & valid_bin
        long_mask = (fl > THRESHOLD).values & valid_bin

        if short_mask.any():
            np.add.at(short_acc, gen_bin[short_mask], w[short_mask])
        if long_mask.any():
            np.add.at(long_acc, gen_bin[long_mask], w[long_mask])

    ratios = np.full(N_WIN, np.nan, dtype=np.float64)
    for wi in range(N_WIN):
        ls = long_acc[wi]
        if ls > 0 and short_acc[wi] + ls >= MIN_WEIGHTED_COUNT:
            ratios[wi] = short_acc[wi] / ls

    group = "healthy" if path.name.endswith("_healthy.csv") else "lung_cancer"
    return path.stem, group, ratios


def _zscore_within_sample(mat: np.ndarray) -> np.ndarray:
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


def _load_or_build(
    length_mode: str, workers: int, unweighted: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    suffix = "_unweighted" if unweighted else ""
    cache = OUT_DIR / f"sl_ratio_training_window_{length_mode}{suffix}.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        print(f"Loaded cache {cache}")
        return data["healthy_sl"], data["cancer_sl"]

    files = sorted(TRAIN_FRAG.glob("*_healthy.csv")) + sorted(
        TRAIN_FRAG.glob("*_lung_cancer.csv")
    )
    print(f"[{length_mode}] Computing s/l from {len(files)} samples ...")
    healthy, cancer = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_sample_window_sl, str(p), length_mode, unweighted): p
            for p in files
        }
        done = 0
        for fut in as_completed(futs):
            _, group, ratios = fut.result()
            (healthy if group == "healthy" else cancer).append(ratios)
            done += 1
            if done % 20 == 0 or done == len(files):
                print(f"  processed {done}/{len(files)} samples", flush=True)

    h_mat = np.vstack(healthy)
    c_mat = np.vstack(cancer)
    np.savez(cache, healthy_sl=h_mat, cancer_sl=c_mat)
    print(f"Wrote cache {cache}")
    return h_mat, c_mat


def _plot_profile(
    h_z: np.ndarray,
    c_z: np.ndarray,
    window_ids: list[str],
    out_png: Path,
    suptitle: str,
    ylabel: str,
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
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(f"{title} (N={n})", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")
        for chrom_i, (xi, _) in enumerate(_chromosome_boundaries(window_ids)):
            if chrom_i > 0:
                ax.axvline(xi - 0.5, color="#888888", linestyle=":", linewidth=0.6, alpha=0.5)

    tick_pos = [m[0] for m in _chromosome_boundaries(window_ids)]
    tick_lbl = [labels[i] for i in tick_pos]
    axes[1].set_xticks(tick_pos)
    axes[1].set_xticklabels(tick_lbl, rotation=45, ha="right", fontsize=7, fontweight="bold")
    for label in axes[1].get_xticklabels():
        label.set_fontweight("bold")

    axes[0].set_ylim(-2, 2)
    axes[1].set_ylim(-2, 2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def _run_one(length_mode: str, workers: int, unweighted: bool = False) -> None:
    weight_label = "unweighted" if unweighted else "weighted"
    unweighted_suffix = "_unweighted" if unweighted else ""
    if length_mode == "100_220":
        out_png = OUT_DIR / f"gc_sl_ratio_by_window_100_220bp{unweighted_suffix}.png"
        suptitle = (
            f"S/L ratio ({weight_label} and z-scored) by 5 Mb window "
            "(100-220bp fragments)"
        )
    else:
        out_png = OUT_DIR / f"gc_sl_ratio_by_window_all_lengths{unweighted_suffix}.png"
        suptitle = f"S/L ratio ({weight_label} and z-scored) by 5 Mb window"

    h_raw, c_raw = _load_or_build(length_mode, workers, unweighted)
    all_raw = np.vstack([h_raw, c_raw])
    keep = np.all(np.isfinite(all_raw), axis=0)
    n_drop = int((~keep).sum())
    print(f"Windows kept: {keep.sum()} / {N_WIN} (dropped {n_drop} with any sample NA)")

    if keep.sum() < MIN_WINDOWS_FOR_Z:
        raise SystemExit(f"Too few windows after filtering ({length_mode})")

    win_ids = [WINDOW_IDS[i] for i in np.where(keep)[0]]
    h_z = _zscore_within_sample(h_raw[:, keep])
    c_z = _zscore_within_sample(c_raw[:, keep])

    _plot_profile(
        h_z,
        c_z,
        win_ids,
        out_png,
        suptitle,
        "Short/long ratio (z-score)",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--mode",
        choices=("both", "100_220", "all"),
        default="both",
        help="Which profile(s) to compute",
    )
    ap.add_argument(
        "--unweighted",
        action="store_true",
        help="Count each fragment as weight 1 (ignore GC weight column)",
    )
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    modes = ["100_220", "all"] if args.mode == "both" else [args.mode]
    for m in modes:
        _run_one(m, args.workers, args.unweighted)


if __name__ == "__main__":
    main()
