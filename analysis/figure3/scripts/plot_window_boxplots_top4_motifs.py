#!/usr/bin/env python3
"""Boxplots for 8 selected windows per top-4 motif (3 sig + 5 nonsig)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

_scripts = Path(__file__).resolve().parents[1] / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from nmb_window_ratios_core import create_genomic_bins  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
CACHE_NPZ = OUT_DIR / "end4motif_top4_per_window_frequencies.npz"
TOP4 = ["CCTG", "CCCA", "CCAG", "CCAA"]
WINDOW_IDS = [b[3] for b in create_genomic_bins(5_000_000)]
WIN_INDEX = {w: i for i, w in enumerate(WINDOW_IDS)}


def _select_windows(csv_path: Path) -> tuple[list[str], list[float]]:
    df = pd.read_csv(csv_path)
    ok = df[df["all_samples_nonzero"]].copy()
    top3 = ok[ok["adj_p_value"] < 0.05].nlargest(3, "delta_median")
    nonsig = ok[ok["adj_p_value"] > 0.05].copy()
    nonsig["abs_delta"] = nonsig["delta_median"].abs()
    bot5 = nonsig.nsmallest(5, "abs_delta")
    sel = pd.concat([top3, bot5], ignore_index=True)
    return sel["window_id"].tolist(), sel["p_value"].tolist()


def _full_window_label(wid: str) -> str:
    parts = wid.split("_")
    if len(parts) < 3:
        return wid
    chrom_disp = parts[0][0].upper() + parts[0][1:]
    start_mb = int(parts[1]) / 1_000_000
    end_mb = int(parts[2]) / 1_000_000

    def _fmt_mb(x: float) -> str:
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.1f}".rstrip("0").rstrip(".")

    return f"{chrom_disp}: {_fmt_mb(start_mb)}-{_fmt_mb(end_mb)}Mb"


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _p_text(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def _draw_bracket(ax, x1: float, x2: float, y: float, h: float, label: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.4, clip_on=False)
    ax.text(
        (x1 + x2) / 2,
        y + h,
        label,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        clip_on=False,
    )


def _plot_motif(
    motif: str,
    motif_idx: int,
    windows: list[str],
    p_values: list[float],
    h_arr: np.ndarray,
    c_arr: np.ndarray,
) -> None:
    rng = np.random.default_rng(42)
    n_h, n_c = h_arr.shape[0], c_arr.shape[0]
    n = len(windows)
    group_gap = 3.0
    box_w = 0.7
    jitter = 0.08

    fig, ax = plt.subplots(figsize=(max(14, n * 1.8), 6.5))
    centers = []

    for i, (wid, pval) in enumerate(zip(windows, p_values)):
        w_idx = WIN_INDEX[wid]
        h_vals = h_arr[:, w_idx, motif_idx]
        c_vals = c_arr[:, w_idx, motif_idx]
        pos_h = i * group_gap
        pos_c = pos_h + 1.0
        centers.append(pos_h + 0.5)

        ax.boxplot(
            h_vals,
            positions=[pos_h],
            widths=box_w,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.8},
            boxprops={"facecolor": "#1f77b4", "edgecolor": "black", "linewidth": 1.4},
            whiskerprops={"color": "black", "linewidth": 1.2},
            capprops={"color": "black", "linewidth": 1.2},
        )
        ax.boxplot(
            c_vals,
            positions=[pos_c],
            widths=box_w,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.8},
            boxprops={"facecolor": "#d62728", "edgecolor": "black", "linewidth": 1.4},
            whiskerprops={"color": "black", "linewidth": 1.2},
            capprops={"color": "black", "linewidth": 1.2},
        )
        ax.scatter(
            pos_h + rng.normal(0, jitter, len(h_vals)),
            h_vals,
            color="#1f77b4",
            edgecolors="black",
            linewidths=0.3,
            s=16,
            alpha=0.65,
            zorder=3,
        )
        ax.scatter(
            pos_c + rng.normal(0, jitter, len(c_vals)),
            c_vals,
            color="#d62728",
            edgecolors="black",
            linewidths=0.3,
            s=16,
            alpha=0.65,
            zorder=3,
        )

        local_max = float(max(h_vals.max(), c_vals.max()))
        bracket_y = min(local_max + 0.18, 4.35)
        stars = _sig_stars(pval)
        label = f"{_p_text(pval)}\n{stars}"
        _draw_bracket(ax, pos_h, pos_c, bracket_y, 0.07, label)

    ax.set_xticks(centers)
    ax.set_xticklabels(
        [_full_window_label(w) for w in windows],
        fontsize=11,
        fontweight="bold",
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("5' end 4-mer motif frequency (%)", fontsize=14, fontweight="bold")
    ax.set_title(motif, fontsize=16, fontweight="bold")
    ax.set_ylim(0, 5)
    ax.tick_params(axis="y", labelsize=12)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.legend(
        handles=[
            Patch(
                facecolor="#1f77b4",
                edgecolor="black",
                label=f"Healthy training samples (N={n_h})",
            ),
            Patch(
                facecolor="#d62728",
                edgecolor="black",
                label=f"Lung cancer training samples (N={n_c})",
            ),
        ],
        loc="upper right",
        prop={"size": 12, "weight": "bold"},
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out_png = OUT_DIR / f"end4motif_{motif}_window_boxplot.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> None:
    if not CACHE_NPZ.exists():
        raise SystemExit(f"Missing cache {CACHE_NPZ}; run window_mannwhitney_top4_motifs.py first")

    data = np.load(CACHE_NPZ, allow_pickle=True)
    h_arr = data["healthy_freqs"]
    c_arr = data["cancer_freqs"]

    for j, motif in enumerate(TOP4):
        csv_path = OUT_DIR / f"end4motif_{motif}_window_mannwhitney.csv"
        windows, p_values = _select_windows(csv_path)
        if len(windows) != 8:
            raise SystemExit(f"{motif}: expected 8 windows, got {len(windows)}")
        _plot_motif(motif, j, windows, p_values, h_arr, c_arr)


if __name__ == "__main__":
    main()
