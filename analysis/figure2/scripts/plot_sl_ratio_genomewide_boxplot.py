#!/usr/bin/env python3
"""
Genome-wide GC-weighted short/long ratio boxplot (training cohort).

- short: sum(weights) for fragments 100–150 bp
- long:  sum(weights) for fragments 151–220 bp
- ratio = short / long per sample; no within-sample z-score
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_scripts = Path(__file__).resolve().parents[1] / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from nmb_window_ratios_core import CHROM_SIZES  # noqa: E402

MATHIOS = Path(__file__).resolve().parents[1]
TRAIN_FRAG = MATHIOS / "fragments_csv" / "training"
OUT_DIR = Path(__file__).resolve().parent
DENSITY_CACHE = OUT_DIR / "fragment_length_genomewide_density_training.npz"

LEN_MIN = 30
SHORT_MIN, SHORT_MAX = 100, 150
LONG_MIN, LONG_MAX = 151, 220
CHUNK = 2_000_000

COLOR_HEALTHY = "#2166AC"
COLOR_CANCER = "#B2182B"
FILL_HEALTHY = "#92C5DE"
FILL_CANCER = "#F4A582"
Y_MAX = 1.0
CANCER_RATIO_CAP = 1.0  # drop lung-cancer samples above this (one extreme outlier)
BOX_LW = 2.2


def _ratio_from_density(dens: np.ndarray) -> float:
    short_w = dens[(SHORT_MIN - LEN_MIN) : (SHORT_MAX - LEN_MIN + 1)].sum()
    long_w = dens[(LONG_MIN - LEN_MIN) : (LONG_MAX - LEN_MIN + 1)].sum()
    return float(short_w / long_w) if long_w > 0 else np.nan


def _sample_genomewide_sl(path_str: str) -> tuple[str, str, float]:
    path = Path(path_str)
    short_sum = 0.0
    long_sum = 0.0

    usecols = ["chromosome", "fragment_length", "weight"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK):
        df = chunk.rename(columns={"chromosome": "chrom"})
        df["chrom"] = df["chrom"].astype(str)
        df["chrom"] = df["chrom"].where(
            df["chrom"].str.startswith("chr"), "chr" + df["chrom"]
        )
        df = df[df["chrom"].isin(CHROM_SIZES.keys())]
        if df.empty:
            continue

        fl = df["fragment_length"].to_numpy(dtype=np.int32, copy=False)
        w = df["weight"].to_numpy(dtype=np.float64, copy=False)
        short_mask = (fl >= SHORT_MIN) & (fl <= SHORT_MAX)
        long_mask = (fl >= LONG_MIN) & (fl <= LONG_MAX)
        if short_mask.any():
            short_sum += float(w[short_mask].sum())
        if long_mask.any():
            long_sum += float(w[long_mask].sum())

    ratio = short_sum / long_sum if long_sum > 0 else np.nan
    group = "healthy" if path.name.endswith("_healthy.csv") else "lung_cancer"
    return path.stem, group, ratio


def _load_from_density_cache() -> tuple[np.ndarray, np.ndarray] | None:
    if not DENSITY_CACHE.exists():
        return None
    data = np.load(DENSITY_CACHE, allow_pickle=True)
    healthy = np.array([_ratio_from_density(d) for d in data["healthy"]], dtype=np.float64)
    cancer = np.array([_ratio_from_density(d) for d in data["cancer"]], dtype=np.float64)
    print(f"Derived S/L from {DENSITY_CACHE}")
    return healthy, cancer


def _load_or_build(workers: int, rebuild: bool) -> tuple[np.ndarray, np.ndarray]:
    if not rebuild:
        cached = _load_from_density_cache()
        if cached is not None:
            return cached

    sl_cache = OUT_DIR / "sl_ratio_genomewide_100_220.npz"
    if sl_cache.exists() and not rebuild:
        data = np.load(sl_cache)
        print(f"Loaded cache {sl_cache}")
        return data["healthy_sl"], data["cancer_sl"]

    files = sorted(TRAIN_FRAG.glob("*_healthy.csv")) + sorted(
        TRAIN_FRAG.glob("*_lung_cancer.csv")
    )
    print(f"Computing genome-wide S/L from {len(files)} samples ...")
    healthy, cancer = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_sample_genomewide_sl, str(p)): p for p in files}
        done = 0
        for fut in as_completed(futs):
            _, group, ratio = fut.result()
            (healthy if group == "healthy" else cancer).append(ratio)
            done += 1
            if done % 20 == 0 or done == len(files):
                print(f"  processed {done}/{len(files)} samples", flush=True)

    healthy = np.asarray(healthy, dtype=np.float64)
    cancer = np.asarray(cancer, dtype=np.float64)
    np.savez(sl_cache, healthy_sl=healthy, cancer_sl=cancer)
    print(f"Wrote cache {sl_cache}")
    return healthy, cancer


def _format_p(p: float) -> str:
    if p < 0.001:
        return f"{p:.4f}"
    return f"{p:.3f}".rstrip("0").rstrip(".")


def _plot_boxplot(healthy: np.ndarray, cancer: np.ndarray, out_png: Path, seed: int) -> None:
    h = healthy[np.isfinite(healthy)]
    c = cancer[np.isfinite(cancer)]
    n_c_before = len(c)
    c = c[c <= CANCER_RATIO_CAP]
    if len(c) < n_c_before:
        print(f"  Excluded {n_c_before - len(c)} lung cancer sample(s) with ratio > {CANCER_RATIO_CAP}")

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    fig, ax = plt.subplots(figsize=(6, 6))
    positions = [1, 2]
    data = [h, c]
    labels = ["Healthy", "Lung cancer"]

    box_kw = dict(
        patch_artist=True,
        widths=0.5,
        showfliers=False,
        boxprops=dict(linewidth=BOX_LW, edgecolor="black"),
        whiskerprops=dict(linewidth=BOX_LW, color="black"),
        capprops=dict(linewidth=BOX_LW, color="black"),
        medianprops=dict(color="black", linewidth=BOX_LW),
    )
    try:
        bp = ax.boxplot(
            data,
            positions=positions,
            tick_labels=labels,
            **box_kw,
        )
    except TypeError:
        bp = ax.boxplot(
            data,
            positions=positions,
            labels=labels,
            **box_kw,
        )

    for patch, fill in zip(bp["boxes"], (FILL_HEALTHY, FILL_CANCER)):
        patch.set_facecolor(fill)
        patch.set_edgecolor("black")
        patch.set_linewidth(BOX_LW)
        patch.set_alpha(0.85)

    rng = np.random.default_rng(seed)
    for pos, vals, color in zip(positions, data, (COLOR_HEALTHY, COLOR_CANCER)):
        jitter = pos + 0.11 * rng.standard_normal(len(vals))
        ax.scatter(jitter, vals, s=18, alpha=0.45, c=color, edgecolors="none", zorder=3)

    if len(h) >= 1 and len(c) >= 1:
        _, p = stats.mannwhitneyu(h, c, alternative="two-sided")
        ax.text(
            0.5,
            0.98,
            f"Mann-Whitney U, p = {_format_p(p)}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylim(0, Y_MAX)
    ax.set_ylabel("Short/long ratio", fontsize=12, fontweight="bold")
    ax.set_title("Genome-wide short/long ratio", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"  Healthy: n={len(h)}, median={np.median(h):.4f}")
    print(f"  Lung cancer: n={len(c)}, median={np.median(c):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--rebuild", action="store_true", help="Re-read fragments (ignore caches)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    healthy, cancer = _load_or_build(args.workers, args.rebuild)

    out_csv = OUT_DIR / "sl_ratio_genomewide_100_220_samples.csv"
    rows = (
        [("healthy", float(v)) for v in healthy if np.isfinite(v)]
        + [("lung_cancer", float(v)) for v in cancer if np.isfinite(v)]
    )
    pd.DataFrame(rows, columns=["group", "short_long_ratio"]).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    out_png = OUT_DIR / "fragment_length_genomewide_sl_ratio_boxplot.png"
    _plot_boxplot(healthy, cancer, out_png, args.seed)


if __name__ == "__main__":
    main()
