#!/usr/bin/env python3
"""
Genome-wide weighted fragment-length density: one curve per training sample.

- All fragments on reference chromosomes (CHROM_SIZES), pooled across the genome.
- Density uses precomputed GC weights in each fragments CSV.
- Healthy = blue, lung cancer = red; all samples overlaid on one panel.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

_scripts = Path(__file__).resolve().parents[1] / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from nmb_window_ratios_core import CHROM_SIZES  # noqa: E402

MATHIOS = Path(__file__).resolve().parents[1]
TRAIN_FRAG = MATHIOS / "fragments_csv" / "training"
OUT_DIR = Path(__file__).resolve().parent
FIGURE2_DIR = MATHIOS.parent / "mFrag_Figures" / "Figure2"

LEN_MIN = 30
LEN_MAX = 500
N_LEN = LEN_MAX - LEN_MIN + 1
CHUNK = 2_000_000
SMOOTH_SIGMA = 1.5

from fragment_length_plot_annotations import annotate_short_long_split  # noqa: E402

COLOR_HEALTHY = "#2166AC"
COLOR_CANCER = "#B2182B"
ZEBRA_LIGHT = "#F4F4F4"
ZEBRA_DARK = "#D8D8D8"
DEFAULT_BIN_WIDTH = 5


def _bin_edges(plot_min: int, plot_max: int, bin_width: int) -> np.ndarray:
    span = plot_max - plot_min
    if span % bin_width != 0:
        raise ValueError(
            f"plot range [{plot_min}, {plot_max}] must be divisible by bin_width={bin_width}"
        )
    return np.arange(plot_min, plot_max + 1, bin_width, dtype=np.float64)


def _draw_zebra_bins(ax, plot_min: int, plot_max: int, bin_width: int) -> list[float]:
    """Alternating vertical stripes per bin; return bin centers for xticks."""
    from matplotlib.transforms import blended_transform_factory

    edges = _bin_edges(plot_min, plot_max, bin_width)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    centers: list[float] = []
    for i in range(len(edges) - 1):
        left = float(edges[i])
        right = float(edges[i + 1])
        color = ZEBRA_LIGHT if i % 2 == 0 else ZEBRA_DARK
        ax.axvspan(
            left,
            right,
            ymin=0,
            ymax=1,
            transform=trans,
            facecolor=color,
            edgecolor="none",
            zorder=0,
        )
        centers.append(0.5 * (left + right))
    return centers


def _set_binned_xticks(ax, plot_min: int, plot_max: int, bin_width: int) -> None:
    edges = _bin_edges(plot_min, plot_max, bin_width)
    ax.set_xticks(edges)
    ax.set_xticklabels([str(int(v)) for v in edges], rotation=45, ha="right", fontsize=8, fontweight="bold")


def _sample_weighted_density(path_str: str) -> tuple[str, str, np.ndarray]:
    path = Path(path_str)
    hist = np.zeros(N_LEN, dtype=np.float64)

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
        ok = (fl >= LEN_MIN) & (fl <= LEN_MAX)
        if not ok.any():
            continue
        idx = (fl[ok] - LEN_MIN).astype(np.intp, copy=False)
        np.add.at(hist, idx, w[ok])

    total = hist.sum()
    if total > 0:
        hist /= total

    group = "healthy" if path.name.endswith("_healthy.csv") else "lung_cancer"
    return path.stem, group, hist


def _load_or_build(workers: int, rebuild: bool) -> tuple[list[np.ndarray], list[np.ndarray]]:
    cache = OUT_DIR / "fragment_length_genomewide_density_training.npz"
    if cache.exists() and not rebuild:
        data = np.load(cache, allow_pickle=True)
        print(f"Loaded cache {cache}")
        return list(data["healthy"]), list(data["cancer"])

    files = sorted(TRAIN_FRAG.glob("*_healthy.csv")) + sorted(
        TRAIN_FRAG.glob("*_lung_cancer.csv")
    )
    print(f"Computing genome-wide weighted densities from {len(files)} samples ...")
    healthy, cancer = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_sample_weighted_density, str(p)): p for p in files}
        done = 0
        for fut in as_completed(futs):
            _, group, dens = fut.result()
            (healthy if group == "healthy" else cancer).append(dens)
            done += 1
            if done % 20 == 0 or done == len(files):
                print(f"  processed {done}/{len(files)} samples", flush=True)

    np.savez(cache, healthy=np.array(healthy), cancer=np.array(cancer))
    print(f"Wrote cache {cache}")
    return healthy, cancer


def _smooth_curve(dens: np.ndarray) -> np.ndarray:
    if dens.sum() <= 0:
        return dens
    y = gaussian_filter1d(dens, sigma=SMOOTH_SIGMA, mode="nearest")
    s = y.sum()
    return y / s if s > 0 else y


def _slice_renorm(dens: np.ndarray, plot_min: int, plot_max: int) -> np.ndarray:
    sub = dens[(plot_min - LEN_MIN) : (plot_max - LEN_MIN + 1)].copy()
    total = sub.sum()
    return sub / total if total > 0 else sub


def _plot_overlay(
    healthy: list[np.ndarray],
    cancer: list[np.ndarray],
    out_png: Path,
    plot_min: int,
    plot_max: int,
    *,
    bin_width: int | None = None,
) -> None:
    x = np.arange(plot_min, plot_max + 1, dtype=np.float64)

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    fig, ax = plt.subplots(figsize=(14, 6) if bin_width else (12, 6))

    ax.set_xlim(plot_min, plot_max)
    use_bins = bin_width is not None and bin_width > 0
    if use_bins:
        _draw_zebra_bins(ax, plot_min, plot_max, bin_width)

    for dens in healthy:
        ax.plot(
            x,
            _smooth_curve(_slice_renorm(dens, plot_min, plot_max)),
            color=COLOR_HEALTHY,
            alpha=0.22,
            linewidth=0.7,
            zorder=1,
        )
    for dens in cancer:
        ax.plot(
            x,
            _smooth_curve(_slice_renorm(dens, plot_min, plot_max)),
            color=COLOR_CANCER,
            alpha=0.22,
            linewidth=0.7,
            zorder=2,
        )

    ax.plot([], [], color=COLOR_HEALTHY, linewidth=2, label=f"Healthy (N={len(healthy)})")
    ax.plot([], [], color=COLOR_CANCER, linewidth=2, label=f"Lung cancer (N={len(cancer)})")

    ax.set_xlabel("Fragment length (bp)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Weighted density", fontsize=12, fontweight="bold")
    ax.set_title(
        "Genome-wide fragment length distribution",
        fontsize=14,
        fontweight="bold",
    )
    if use_bins:
        _set_binned_xticks(ax, plot_min, plot_max, bin_width)
        fig.subplots_adjust(bottom=0.18)
    ax.legend(loc="upper right", frameon=True, fontsize=11)
    ax.grid(True, alpha=0.25, axis="y")
    annotate_short_long_split(ax, plot_min, plot_max)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--rebuild", action="store_true", help="Ignore NPZ cache")
    ap.add_argument("--plot-min", type=int, default=LEN_MIN)
    ap.add_argument("--plot-max", type=int, default=LEN_MAX)
    ap.add_argument(
        "--bin-width",
        type=int,
        default=0,
        help="If >0, draw zebra stripes and x ticks per bin (bp); range must be divisible",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: overlay or overlay_{min}_{max}.png)",
    )
    args = ap.parse_args()
    if args.plot_min < LEN_MIN or args.plot_max > LEN_MAX:
        raise SystemExit(f"plot range must lie within [{LEN_MIN}, {LEN_MAX}]")
    if args.plot_min >= args.plot_max:
        raise SystemExit("--plot-min must be < --plot-max")

    bin_width = args.bin_width
    if bin_width <= 0 and (args.plot_min, args.plot_max) == (100, 220):
        bin_width = DEFAULT_BIN_WIDTH
    if bin_width > 0 and (args.plot_max - args.plot_min) % bin_width != 0:
        raise SystemExit(
            f"plot range must be divisible by --bin-width={bin_width}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    healthy, cancer = _load_or_build(args.workers, args.rebuild)
    if args.out is not None:
        out_png = args.out
    elif (args.plot_min, args.plot_max) == (100, 220):
        FIGURE2_DIR.mkdir(parents=True, exist_ok=True)
        out_png = FIGURE2_DIR / "fragment_length_genomewide_density_overlay_100_220.png"
    elif (args.plot_min, args.plot_max) == (LEN_MIN, LEN_MAX):
        out_png = OUT_DIR / "fragment_length_genomewide_density_overlay.png"
    else:
        out_png = (
            OUT_DIR
            / f"fragment_length_genomewide_density_overlay_{args.plot_min}_{args.plot_max}.png"
        )
    _plot_overlay(
        healthy,
        cancer,
        out_png,
        args.plot_min,
        args.plot_max,
        bin_width=bin_width if bin_width > 0 else None,
    )


if __name__ == "__main__":
    main()
