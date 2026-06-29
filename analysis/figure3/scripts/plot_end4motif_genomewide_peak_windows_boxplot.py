#!/usr/bin/env python3
"""Motif boxplots: genome-wide vs top-2 per-window -log10(p) peaks."""
from __future__ import annotations

import argparse
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
from plot_end4motif_top_bottom_boxplots import (  # noqa: E402
    CACHE_NPZ as GW_CACHE,
    MOTIF_INDEX,
    MW_CSV,
    _draw_bracket,
    _p_text,
)
from plot_motif_window_median_difference import (  # noqa: E402
    N_TOP_PEAKS,
    _neg_log10_p,
    _per_window_pvalues,
)
from plot_window_boxplots_top4_motifs import TOP4, _full_window_label  # noqa: E402
from window_mannwhitney_top4_motifs import CACHE_NPZ as WIN_CACHE  # noqa: E402

HERE = Path(__file__).resolve().parent
CFCRAFT = HERE.parents[1]
FIGURE3_DIR = CFCRAFT / "mFrag_Figures" / "Figure3"
WINDOW_IDS = [b[3] for b in create_genomic_bins(5_000_000)]
WIN_INDEX = {w: i for i, w in enumerate(WINDOW_IDS)}


def _peak_window_ids(h_arr: np.ndarray, c_arr: np.ndarray) -> list[str]:
    neg_log_p = _neg_log10_p(_per_window_pvalues(h_arr, c_arr))
    valid = np.isfinite(neg_log_p)
    top_idx = np.argsort(np.where(valid, neg_log_p, -np.inf))[-N_TOP_PEAKS:]
    top_idx = top_idx[np.argsort(neg_log_p[top_idx])[::-1]]
    return [WINDOW_IDS[i] for i in top_idx]


def _load_p_values(motif: str, window_ids: list[str]) -> list[float]:
    gw = pd.read_csv(MW_CSV)
    row = gw.loc[gw["motif"] == motif]
    if row.empty:
        raise SystemExit(f"{motif} not found in {MW_CSV}")
    pvals = [float(row.iloc[0]["p_value"])]

    win_csv = HERE / f"end4motif_{motif}_window_mannwhitney.csv"
    win = pd.read_csv(win_csv).set_index("window_id")
    for wid in window_ids:
        if wid not in win.index:
            raise SystemExit(f"Missing {wid} in {win_csv}")
        pvals.append(float(win.loc[wid, "p_value"]))
    return pvals


def _bracket_y(label: str, local_max: float, ylim_top: float) -> float:
    if label.startswith("Chr7") and "55-60" in label:
        return 2.92
    return min(local_max + 0.12, ylim_top - 0.45)


def _plot_boxplots(
    motif: str,
    labels: list[str],
    p_values: list[float],
    h_groups: list[np.ndarray],
    c_groups: list[np.ndarray],
    *,
    ylim: tuple[float, float],
    out_png: Path,
    out_pdf: Path,
) -> None:
    rng = np.random.default_rng(42)
    n_h, n_c = h_groups[0].shape[0], c_groups[0].shape[0]
    n = len(labels)
    group_gap = 3.0
    box_w = 0.7
    jitter = 0.08
    ylim_top = ylim[1]

    fig, ax = plt.subplots(figsize=(max(10, n * 2.2), 6))
    centers = []

    for label, pval, h_vals, c_vals in zip(labels, p_values, h_groups, c_groups):
        pos_h = len(centers) * group_gap
        pos_c = pos_h + 1.0
        centers.append(pos_h + 0.5)

        ax.boxplot(
            h_vals,
            positions=[pos_h],
            widths=box_w,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
            boxprops={"facecolor": "#1f77b4", "edgecolor": "black", "linewidth": 1.2},
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
        )
        ax.boxplot(
            c_vals,
            positions=[pos_c],
            widths=box_w,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
            boxprops={"facecolor": "#d62728", "edgecolor": "black", "linewidth": 1.2},
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
        )
        ax.scatter(
            pos_h + rng.normal(0, jitter, len(h_vals)),
            h_vals,
            color="#1f77b4",
            edgecolors="black",
            linewidths=0.3,
            s=14,
            alpha=0.65,
            zorder=3,
        )
        ax.scatter(
            pos_c + rng.normal(0, jitter, len(c_vals)),
            c_vals,
            color="#d62728",
            edgecolors="black",
            linewidths=0.3,
            s=14,
            alpha=0.65,
            zorder=3,
        )

        local_max = float(max(h_vals.max(), c_vals.max()))
        bracket_y = _bracket_y(label, local_max, ylim_top)
        _draw_bracket(ax, pos_h, pos_c, bracket_y, 0.06, _p_text(pval))

    ax.set_xticks(centers)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel(f"{motif} motif frequency (%)", fontsize=12, fontweight="bold")
    ax.set_title(motif, fontsize=14, fontweight="bold")
    ax.set_ylim(*ylim)
    ax.tick_params(axis="y", labelsize=11)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.legend(
        handles=[
            Patch(facecolor="#1f77b4", edgecolor="black", label=f"Healthy (N={n_h})"),
            Patch(facecolor="#d62728", edgecolor="black", label=f"Lung cancer (N={n_c})"),
        ],
        loc="upper right",
        prop={"size": 11, "weight": "bold"},
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motif", default="CCCA", choices=["CCTG", "CCCA"])
    ap.add_argument("--ylim-min", type=float, default=1.0)
    ap.add_argument("--ylim-max", type=float, default=3.5)
    ap.add_argument("--out-png", type=Path, default=None)
    ap.add_argument("--out-pdf", type=Path, default=None)
    args = ap.parse_args()

    motif = args.motif
    if motif not in TOP4:
        raise SystemExit(f"{motif} must be in {TOP4}")
    motif_idx = TOP4.index(motif)

    out_png = args.out_png or FIGURE3_DIR / f"end4motif_{motif}_genomewide_peak_windows_boxplot.png"
    out_pdf = args.out_pdf or FIGURE3_DIR / f"end4motif_{motif}_genomewide_peak_windows_boxplot.pdf"

    if not GW_CACHE.exists():
        raise SystemExit(f"Missing {GW_CACHE}")
    if not WIN_CACHE.exists():
        raise SystemExit(f"Missing {WIN_CACHE}; run window_mannwhitney_top4_motifs.py first")

    gw = np.load(GW_CACHE, allow_pickle=True)
    h_gw = gw["healthy_freqs"][:, MOTIF_INDEX[motif]]
    c_gw = gw["cancer_freqs"][:, MOTIF_INDEX[motif]]

    win = np.load(WIN_CACHE, allow_pickle=True)
    h_win = win["healthy_freqs"]
    c_win = win["cancer_freqs"]

    peak_ids = _peak_window_ids(h_win[:, :, motif_idx], c_win[:, :, motif_idx])
    for wid in peak_ids:
        print(f"Peak window: {wid} ({_full_window_label(wid)})")

    labels = ["Genome-wide", *[_full_window_label(w) for w in peak_ids]]
    p_values = _load_p_values(motif, peak_ids)

    h_groups = [h_gw]
    c_groups = [c_gw]
    for wid in peak_ids:
        w_idx = WIN_INDEX[wid]
        h_groups.append(h_win[:, w_idx, motif_idx])
        c_groups.append(c_win[:, w_idx, motif_idx])

    _plot_boxplots(
        motif,
        labels,
        p_values,
        h_groups,
        c_groups,
        ylim=(args.ylim_min, args.ylim_max),
        out_png=out_png,
        out_pdf=out_pdf,
    )


if __name__ == "__main__":
    main()
