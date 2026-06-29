#!/usr/bin/env python3
"""Genome-wide stacked-bar profile of 4-mer motif group median frequencies."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_end4motif_top_bottom_boxplots import (
    CACHE_NPZ,
    MOTIFS,
    OUT_DIR,
    _load_or_build_sample_matrix,
)

DEFAULT_OUT_PNG = OUT_DIR / "end4motif_training_genomewide_frequency_curves.png"
DEFAULT_OUT_PDF = OUT_DIR / "end4motif_training_genomewide_frequency_curves.pdf"
COLOR_HEALTHY = "#1f77b4"
COLOR_CANCER = "#d62728"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=CACHE_NPZ)
    p.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    p.add_argument("--out-pdf", type=Path, default=DEFAULT_OUT_PDF)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument(
        "--tick-step",
        type=int,
        default=8,
        help="Label every Nth motif on x-axis (default: 8 → 32 labels)",
    )
    p.add_argument(
        "--sort-by",
        choices=("total", "healthy", "cancer", "lexicographic"),
        default="total",
        help="X-axis order: total/healthy/cancer median frequency (high→low) or lexicographic",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.cache != CACHE_NPZ and args.cache.exists():
        data = np.load(args.cache, allow_pickle=True)
        h_mat = data["healthy_freqs"]
        c_mat = data["cancer_freqs"]
        print(f"Loaded cache {args.cache}")
    else:
        h_mat, c_mat, _, _ = _load_or_build_sample_matrix(args.workers)

    h_med = np.median(h_mat, axis=0)
    c_med = np.median(c_mat, axis=0)

    if args.sort_by == "lexicographic":
        order = np.arange(256)
        xlabel = "5$'$ end 4-mer motif (lexicographic ACGT order)"
    else:
        if args.sort_by == "healthy":
            sort_key = h_med
            xlabel = "5$'$ end 4-mer motif (healthy median frequency, high→low)"
        elif args.sort_by == "cancer":
            sort_key = c_med
            xlabel = "5$'$ end 4-mer motif (lung cancer median frequency, high→low)"
        else:
            sort_key = h_med + c_med
            xlabel = "5$'$ end 4-mer motif (total median frequency, high→low)"
        order = np.argsort(-sort_key, kind="stable")

    h_med = h_med[order]
    c_med = c_med[order]
    motifs_sorted = [MOTIFS[i] for i in order]
    x = np.arange(256)

    plt.rcParams.update(
        {
            "font.size": 12,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
        }
    )

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(
        x,
        h_med,
        width=1.0,
        color=COLOR_HEALTHY,
        edgecolor="none",
        align="center",
        label=f"Healthy median (N={h_mat.shape[0]})",
        zorder=2,
    )
    ax.bar(
        x,
        c_med,
        width=1.0,
        bottom=h_med,
        color=COLOR_CANCER,
        edgecolor="none",
        align="center",
        label=f"Lung cancer median (N={c_mat.shape[0]})",
        zorder=2,
    )

    step = max(1, args.tick_step)
    tick_pos = np.arange(0, 256, step)
    tick_lbl = [motifs_sorted[i] for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(
        tick_lbl,
        rotation=45,
        ha="right",
        va="top",
        fontsize=7,
        fontweight="bold",
    )
    ax.tick_params(axis="x", pad=10)

    ax.set_xlim(-0.5, 255.5)
    ax.set_xlabel(xlabel, labelpad=12)
    ax.set_ylabel("Genome-wide motif frequency (%)")
    ax.set_title(
        "Genome-wide 5$'$ end 4-mer motif frequency (stacked group medians)"
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.legend(loc="upper right", framealpha=0.9, prop={"size": 11, "weight": "bold"})

    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.32)
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=300)
    fig.savefig(args.out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out_png}")
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
