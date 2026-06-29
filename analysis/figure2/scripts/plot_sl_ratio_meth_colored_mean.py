#!/usr/bin/env python3
"""GC s/l ratio by window: grey samples + meth β-colored LOESS group mean."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_scripts = Path(__file__).resolve().parents[1] / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from meth_beta_colored_loess_mean import plot_two_panel_meth_mean  # noqa: E402
from plot_sl_ratio_by_window import (  # noqa: E402
    MIN_WINDOWS_FOR_Z,
    WINDOW_IDS,
    _chromosome_boundaries,
    _window_label,
    _zscore_within_sample,
)

OUT_DIR = Path(__file__).resolve().parent

def _mode_config(length_mode: str, unweighted: bool) -> dict:
    suffix = "_unweighted" if unweighted else ""
    weight_label = "unweighted" if unweighted else "weighted"
    if length_mode == "100_220":
        return {
            "cache_key": "100_220",
            "out_png": OUT_DIR / f"gc_sl_ratio_by_window_100_220bp_meth_beta_mean{suffix}.png",
            "suptitle": (
                f"S/L ratio ({weight_label} and z-scored) by 5 Mb window "
                "(100-220bp fragments)"
            ),
        }
    return {
        "cache_key": "all",
        "out_png": OUT_DIR / f"gc_sl_ratio_by_window_all_lengths_meth_beta_mean{suffix}.png",
        "suptitle": f"S/L ratio ({weight_label} and z-scored) by 5 Mb window",
    }


def _run_mode(
    length_mode: str, unweighted: bool = False
) -> tuple[list[str], np.ndarray, np.ndarray, Path, str]:
    cfg = _mode_config(length_mode, unweighted)
    cache_suffix = "_unweighted" if unweighted else ""
    cache = OUT_DIR / f"sl_ratio_training_window_{cfg['cache_key']}{cache_suffix}.npz"
    if not cache.exists():
        raise SystemExit(f"Missing cache {cache}; run plot_sl_ratio_by_window.py first")

    data = np.load(cache, allow_pickle=True)
    h_raw, c_raw = data["healthy_sl"], data["cancer_sl"]
    all_raw = np.vstack([h_raw, c_raw])
    keep = np.all(np.isfinite(all_raw), axis=0)
    win_ids = [WINDOW_IDS[i] for i in np.where(keep)[0]]
    if len(win_ids) < MIN_WINDOWS_FOR_Z:
        raise SystemExit(f"Too few windows after filtering ({length_mode})")

    h_z = _zscore_within_sample(h_raw[:, keep])
    c_z = _zscore_within_sample(c_raw[:, keep])
    return win_ids, h_z, c_z, cfg["out_png"], cfg["suptitle"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("both", "100_220", "all"), default="both")
    ap.add_argument("--loess-frac", type=float, default=0.08)
    ap.add_argument(
        "--unweighted",
        action="store_true",
        help="Use unweighted s/l cache (each fragment weight 1)",
    )
    args = ap.parse_args()

    modes = ["100_220", "all"] if args.mode == "both" else [args.mode]
    for m in modes:
        win_ids, h_z, c_z, out_png, suptitle = _run_mode(m, args.unweighted)
        all_mat = np.vstack([h_z, c_z])
        global_mean = np.nanmean(all_mat, axis=0)
        plot_two_panel_meth_mean(
            h_z - global_mean,
            c_z - global_mean,
            win_ids,
            ylabel="Short/long ratio (z-scored)",
            suptitle=suptitle,
            out_png=out_png,
            chromosome_boundaries_fn=_chromosome_boundaries,
            window_label_fn=_window_label,
            loess_frac=args.loess_frac,
        )


if __name__ == "__main__":
    main()
