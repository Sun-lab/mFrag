#!/usr/bin/env python3
"""
Barplot-selected 6 samples: group-mean LOWESS S/L curves (3 groups) vs training
lung-cancer mean (black LOWESS).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

FRAG_LEN_DIR = Path("/fh/working/sun_w/saishi_cui/cfCraft/Mathios/Fragment_length")
MATHIOS_SCRIPTS = Path("/fh/working/sun_w/saishi_cui/cfCraft/Mathios/scripts")
TEST_FRAG = Path("/fh/working/sun_w/saishi_cui/cfCraft/Mathios/fragments_csv/testing")
OUT_DIR = Path("/fh/working/sun_w/saishi_cui/cfCraft/mFrag_Figures/Figure5")
BARPLOT_CSV = OUT_DIR / "figure5_frag_length_stage_II_mfragnet_top2_vs_fp_healthy_predicted_prob_barplot.csv"

if str(MATHIOS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MATHIOS_SCRIPTS))
if str(FRAG_LEN_DIR) not in sys.path:
    sys.path.insert(0, str(FRAG_LEN_DIR))

from meth_beta_colored_loess_mean import (  # noqa: E402
    loess_smooth,
    plot_black_loess_mean,
)
from plot_sl_ratio_by_window import (  # noqa: E402
    MIN_WINDOWS_FOR_Z,
    WINDOW_IDS,
    _chromosome_boundaries,
    _sample_window_sl,
    _window_label,
    _zscore_within_sample,
)

GROUP_STYLE = {
    "stage_II_top_mfragnet": {
        "color": "#ff7f0e",
        "label": "Stage II cancer mean (n=2)",
    },
    "healthy_fp": {
        "color": "#2166AC",
        "label": "Healthy: DELFI high, mFrag-Net low (n=2)",
    },
    "healthy_both_low": {
        "color": "#2ca02c",
        "label": "Healthy: both low (n=2)",
    },
}

GROUP_ORDER = ("stage_II_top_mfragnet", "healthy_fp", "healthy_both_low")

LOESS_FRAC = 0.08
GROUP_LINE_WIDTH = 2.8
CANCER_LINE_WIDTH = 3.0
FONT_LABEL = 15
FONT_TICK_X = 10
FONT_TICK_Y = 13
FONT_LEGEND = 11


def _load_training_processed() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache = FRAG_LEN_DIR / "sl_ratio_training_window_all.npz"
    if not cache.is_file():
        raise SystemExit(f"Missing cache {cache}; run plot_sl_ratio_by_window.py first.")
    data = np.load(cache, allow_pickle=True)
    h_raw, c_raw = data["healthy_sl"], data["cancer_sl"]
    all_raw = np.vstack([h_raw, c_raw])
    keep = np.all(np.isfinite(all_raw), axis=0)
    win_ids = [WINDOW_IDS[i] for i in np.where(keep)[0]]
    if len(win_ids) < MIN_WINDOWS_FOR_Z:
        raise SystemExit("Too few windows after filtering.")
    h_z = _zscore_within_sample(h_raw[:, keep])
    c_z = _zscore_within_sample(c_raw[:, keep])
    global_mean = np.nanmean(np.vstack([h_z, c_z]), axis=0)
    return win_ids, h_z - global_mean, c_z - global_mean, global_mean, keep


def _sample_curve(sample_id: str, keep: np.ndarray) -> np.ndarray:
    path = TEST_FRAG / f"{sample_id}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    _, _, ratios = _sample_window_sl(str(path), "all", unweighted=False)
    row_z = _zscore_within_sample(ratios[keep][None, :])
    return row_z[0]


def _load_grouped_curves(
    keep: np.ndarray,
    global_mean: np.ndarray,
) -> dict[str, np.ndarray]:
    df = pd.read_csv(BARPLOT_CSV)
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for _, r in df.iterrows():
        sid = str(r["sample_id"])
        selection = str(r["selection"])
        z = _sample_curve(sid, keep)
        grouped[selection].append(z - global_mean)

    out: dict[str, np.ndarray] = {}
    for key in GROUP_ORDER:
        rows = grouped.get(key, [])
        if len(rows) != 2:
            raise SystemExit(f"Expected 2 samples for {key}, found {len(rows)}")
        out[key] = np.nanmean(np.vstack(rows), axis=0)
    return out


def _plot_group_loess(
    ax,
    x: np.ndarray,
    mean_vals: np.ndarray,
    *,
    color: str,
    label: str,
    zorder: int,
) -> Line2D:
    sm = loess_smooth(x, mean_vals, frac=LOESS_FRAC)
    if sm.shape[0] >= 2:
        ax.plot(
            sm[:, 0],
            sm[:, 1],
            color=color,
            linewidth=GROUP_LINE_WIDTH,
            solid_capstyle="round",
            label=label,
            zorder=zorder,
        )
    return Line2D([0], [0], color=color, linewidth=GROUP_LINE_WIDTH, label=label)


def plot_sl_ratio(
    group_means: dict[str, np.ndarray],
    *,
    out_png: Path,
    out_pdf: Path,
    dpi: int,
) -> None:
    win_ids, _h_plot, c_plot, _global_mean, _keep = _load_training_processed()
    cancer_mean = np.nanmean(c_plot, axis=0)
    x = np.arange(len(win_ids))
    labels = [_window_label(w) for w in win_ids]

    fig, ax = plt.subplots(figsize=(18, 5.5))

    group_handles: list[Line2D] = []
    for i, key in enumerate(GROUP_ORDER):
        style = GROUP_STYLE[key]
        handle = _plot_group_loess(
            ax,
            x,
            group_means[key],
            color=style["color"],
            label=style["label"],
            zorder=8 + i,
        )
        group_handles.append(handle)

    plot_black_loess_mean(
        ax,
        x,
        cancer_mean,
        loess_frac=LOESS_FRAC,
        linewidth=CANCER_LINE_WIDTH,
        zorder=12,
    )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.set_ylabel("Short/long ratio (z-scored)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylim(-2.0, 2.0)
    ax.grid(True, alpha=0.2, axis="y")
    for chrom_i, (xi, _) in enumerate(_chromosome_boundaries(win_ids)):
        if chrom_i > 0:
            ax.axvline(xi - 0.5, color="#888888", linestyle=":", linewidth=0.6, alpha=0.5)

    tick_pos = [m[0] for m in _chromosome_boundaries(win_ids)]
    tick_lbl = [labels[i] for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=45, ha="right", fontsize=FONT_TICK_X, fontweight="bold")
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")

    cancer_handle = Line2D(
        [0],
        [0],
        color="black",
        linewidth=CANCER_LINE_WIDTH,
        label="Lung cancer mean (training)",
    )
    ax.legend(
        handles=group_handles + [cancer_handle],
        loc="upper right",
        framealpha=0.92,
        prop={"weight": "bold", "size": FONT_LEGEND},
    )

    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.22)
    for out in (out_png, out_pdf):
        fig.savefig(out, dpi=dpi)
        print(f"Wrote {out}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    if not BARPLOT_CSV.is_file():
        raise SystemExit(f"Missing {BARPLOT_CSV}; run barplot script first.")

    plt.rcParams.update(
        {
            "font.size": FONT_TICK_Y,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.labelsize": FONT_LABEL,
            "xtick.labelsize": FONT_TICK_X,
            "ytick.labelsize": FONT_TICK_Y,
        }
    )

    _, _, _, global_mean, keep = _load_training_processed()
    group_means = _load_grouped_curves(keep, global_mean)

    stem = "figure5_stage_II_barplot6_sl_ratio_vs_cancer_mean"
    plot_sl_ratio(
        group_means,
        out_png=args.out_dir / f"{stem}.png",
        out_pdf=args.out_dir / f"{stem}.pdf",
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
