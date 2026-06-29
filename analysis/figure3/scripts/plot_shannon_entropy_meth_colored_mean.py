#!/usr/bin/env python3
"""Shannon entropy by window: grey samples + meth β-colored LOESS group mean."""
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
OUT_PNG = OUT_DIR / "end4motif_training_shannon_entropy_by_window_meth_beta_mean.png"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1, help="Only used if cache missing")
    ap.add_argument("--loess-frac", type=float, default=0.08)
    args = ap.parse_args()

    h_raw, c_raw = _load_or_build(args.workers)
    all_raw = np.vstack([h_raw, c_raw])
    keep = np.all(np.isfinite(all_raw), axis=0)
    win_ids = [WINDOW_IDS[i] for i in np.where(keep)[0]]
    if len(win_ids) < MIN_WINDOWS_FOR_Z:
        raise SystemExit("Too few windows after filtering")

    h_z = _zscore_within_sample(h_raw[:, keep])
    c_z = _zscore_within_sample(c_raw[:, keep])
    all_mat = np.vstack([h_z, c_z])
    global_mean = np.nanmean(all_mat, axis=0)

    plot_two_panel_meth_mean(
        h_z - global_mean,
        c_z - global_mean,
        win_ids,
        ylabel="Shannon entropy (z-score)",
        suptitle="5' end 4-mer Shannon entropy by 5 Mb window",
        out_png=OUT_PNG,
        chromosome_boundaries_fn=_chromosome_boundaries,
        window_label_fn=_window_label,
        loess_frac=args.loess_frac,
    )


if __name__ == "__main__":
    main()
