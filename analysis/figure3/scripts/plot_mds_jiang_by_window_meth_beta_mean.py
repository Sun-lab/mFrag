#!/usr/bin/env python3
"""Per-window Jiang MDS (normalized Shannon entropy) with meth β-colored LOESS mean.

MDS per 5 Mb window (Jiang et al. 2020):
  MDS = sum_i [-P_i * log(P_i)] / log(256),  natural log, P_i = motif frequency.

Equivalent to dividing per-window log2 Shannon entropy by log2(256) when using the
same weighted 4-mer counts as plot_shannon_entropy_by_window.py.

Pipeline: raw MDS per window → within-sample z-score → subtract global mean curve → plot.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_scripts = Path(__file__).resolve().parents[1] / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from meth_beta_colored_loess_mean import plot_two_panel_meth_mean  # noqa: E402
from plot_shannon_entropy_by_window import (  # noqa: E402
    MIN_WINDOWS_FOR_Z,
    WINDOW_IDS,
    _chromosome_boundaries,
    _load_or_build,
    _window_label,
    _zscore_within_sample,
)

OUT_DIR = Path(__file__).resolve().parent
CFCRAFT = OUT_DIR.parents[1]
FIGURE3_DIR = CFCRAFT / "mFrag_Figures" / "Figure3"
OUT_PNG = FIGURE3_DIR / "end4motif_training_mds_jiang_by_window_meth_beta_mean.png"

# Jiang 2020: divide normalized Shannon entropy by log(256); same factor for log2 H.
LOG2_256 = float(np.log2(256))


def _shannon_log2_to_jiang_mds(shannon_log2: np.ndarray) -> np.ndarray:
    """Convert per-window -sum p log2(p) to Jiang MDS = -sum p ln(p) / ln(256)."""
    return shannon_log2 / LOG2_256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1, help="Only used if Shannon cache missing")
    ap.add_argument("--loess-frac", type=float, default=0.08)
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT_PNG,
        help="Output PNG path",
    )
    args = ap.parse_args()

    h_sh, c_sh = _load_or_build(args.workers)
    h_raw = _shannon_log2_to_jiang_mds(h_sh)
    c_raw = _shannon_log2_to_jiang_mds(c_sh)

    all_raw = np.vstack([h_raw, c_raw])
    keep = np.all(np.isfinite(all_raw), axis=0)
    win_ids = [WINDOW_IDS[i] for i in np.where(keep)[0]]
    if len(win_ids) < MIN_WINDOWS_FOR_Z:
        raise SystemExit("Too few windows after filtering")

    h_z = _zscore_within_sample(h_raw[:, keep])
    c_z = _zscore_within_sample(c_raw[:, keep])
    global_mean = np.nanmean(np.vstack([h_z, c_z]), axis=0)

    plot_two_panel_meth_mean(
        h_z - global_mean,
        c_z - global_mean,
        win_ids,
        ylabel="Motif Diversity Score (z-scored)",
        suptitle="5' end 4-mer Motif Diversity Score by 5 Mb window",
        out_png=args.out,
        chromosome_boundaries_fn=_chromosome_boundaries,
        window_label_fn=_window_label,
        loess_frac=args.loess_frac,
    )


if __name__ == "__main__":
    main()
