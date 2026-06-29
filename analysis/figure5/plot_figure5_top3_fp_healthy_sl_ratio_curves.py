#!/usr/bin/env python3
"""
Top-3 DELFI-high / mFrag-low healthy testing samples: S/L ratio curves vs
training lung-cancer mean (methylation Δβ-colored), matching
gc_sl_ratio_by_window_all_lengths_meth_beta_mean.png processing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

FRAG_LEN_DIR = Path("/fh/working/sun_w/saishi_cui/cfCraft/Mathios/Fragment_length")
MATHIOS_SCRIPTS = Path("/fh/working/sun_w/saishi_cui/cfCraft/Mathios/scripts")
TEST_FRAG = Path("/fh/working/sun_w/saishi_cui/cfCraft/Mathios/fragments_csv/testing")
OUT_DIR = Path("/fh/working/sun_w/saishi_cui/cfCraft/mFrag_Figures/Figure5")

if str(MATHIOS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MATHIOS_SCRIPTS))
if str(FRAG_LEN_DIR) not in sys.path:
    sys.path.insert(0, str(FRAG_LEN_DIR))

from meth_beta_colored_loess_mean import (  # noqa: E402
    add_meth_colorbar,
    load_meth_mean_difference,
    loess_smooth,
    _segment_color,
)
from plot_sl_ratio_by_window import (  # noqa: E402
    MIN_WINDOWS_FOR_Z,
    WINDOW_IDS,
    _chromosome_boundaries,
    _sample_window_sl,
    _window_label,
    _zscore_within_sample,
)

GROUP_SAMPLES = {
    "healthy_fp": (
        ("CGPLH676P_1_healthy", "CGPLH676P_1"),
        ("CGPLH969P_healthy", "CGPLH969P"),
        ("CGPLH919P_healthy", "CGPLH919P"),
    ),
    "healthy_both_low": (
        ("CGPLH1028P_healthy", "CGPLH1028P"),
        ("CGPLH1019P_healthy", "CGPLH1019P"),
        ("CGPLH1011P_healthy", "CGPLH1011P"),
    ),
}
GROUP_STYLE = {
    "healthy_fp": {
        "color": "#000000",
        "label": "DELFI high, mFrag-Weight low (CGPLH676P_1, CGPLH969P, CGPLH919P)",
    },
    "healthy_both_low": {
        "color": "#2ca02c",
        "label": "Both low (CGPLH1028P, CGPLH1019P, CGPLH1011P)",
    },
}
GROUP_ORDER = ("healthy_fp", "healthy_both_low")
SAMPLE_LINE_WIDTH = 2.8
LOESS_FRAC = 0.08
PANEL_Y_LIMS = {
    "chr1": (-5.0, 4.0),
    "chr4_5": (-2.0, 4.0),
}
PANEL_Y_REF_LINES = {
    "chr1": (-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0),
    "chr4_5": (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0),
}
CANCER_LINE_WIDTH = 4.0
FONT_LABEL = 15
FONT_TICK_X = 10
FONT_TICK_Y = 13
FONT_LEGEND = 12
GAP_WIDTH = 0.35
X_TICK_STEP = 10  # every 10 windows (50 Mb)
PANEL_FACE_COLOR = "#b5b5b5"
CHR1_ZOOM_START_MB = 50_000_000
CHR1_ZOOM_END_MB = 215_000_000
FOCUS_REGIONS = (
    {"key": "chr1", "label": "Chr1"},
    {"key": "chr4_5", "label": "Chr4 (second half) – Chr5 (first half)"},
)


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


def _region_indices(win_ids: list[str], region_key: str) -> list[int]:
    if region_key == "chr1":
        return [
            i
            for i, w in enumerate(win_ids)
            if w.startswith("chr1_")
            and CHR1_ZOOM_START_MB <= int(w.split("_")[1]) <= CHR1_ZOOM_END_MB
        ]
    if region_key == "chr10":
        return [i for i, w in enumerate(win_ids) if w.startswith("chr10_")]
    if region_key == "chr4_5":
        c4 = [i for i, w in enumerate(win_ids) if w.startswith("chr4_")]
        c5 = [i for i, w in enumerate(win_ids) if w.startswith("chr5_")]
        return c4[len(c4) // 2 :] + c5[: len(c5) // 2]
    raise ValueError(f"Unknown region: {region_key}")


def _style_region_xaxis(ax, indices: list[int], win_ids: list[str]) -> None:
    sub_ids = [win_ids[i] for i in indices]
    labels = [_window_label(w) for w in sub_ids]
    boundaries = _chromosome_boundaries(sub_ids)
    n = len(indices)

    ax.set_facecolor(PANEL_FACE_COLOR)
    for chrom_i, (xi, _) in enumerate(boundaries):
        if chrom_i > 0:
            ax.axvline(xi - 0.5, color="#666666", linestyle=":", linewidth=0.6, alpha=0.45)

    tick_pos = list(range(0, n, X_TICK_STEP))
    if n - 1 not in tick_pos:
        tick_pos.append(n - 1)
    tick_lbl = [labels[i] for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=45, ha="right", fontsize=FONT_TICK_X, fontweight="bold")
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")
    ax.set_xlim(-0.5, n - 0.5)


def _precompute_loess(vals: np.ndarray) -> np.ndarray:
    x_full = np.arange(len(vals), dtype=float)
    return loess_smooth(x_full, vals, frac=LOESS_FRAC)


def _plot_loess_region(
    ax,
    sm: np.ndarray,
    indices: list[int],
    *,
    color: str,
    zorder: int,
    linewidth: float = SAMPLE_LINE_WIDTH,
) -> None:
    if sm.shape[0] < 2:
        return
    i0, i1 = indices[0], indices[-1]
    index_set = set(indices)
    x_s, y_s = sm[:, 0], sm[:, 1]
    for i in range(len(x_s) - 1):
        wi = int(round(0.5 * (x_s[i] + x_s[i + 1])))
        if wi not in index_set:
            continue
        ax.plot(
            [x_s[i] - i0, x_s[i + 1] - i0],
            [y_s[i], y_s[i + 1]],
            color=color,
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=zorder,
        )


def _plot_meth_colored_loess_region(
    ax,
    sm: np.ndarray,
    indices: list[int],
    win_ids: list[str],
    meth_map: dict[str, float],
    norm,
    cmap,
    *,
    linewidth: float = CANCER_LINE_WIDTH,
    zorder: int = 20,
) -> None:
    if sm.shape[0] < 2:
        return
    i0 = indices[0]
    index_set = set(indices)
    x_s, y_s = sm[:, 0], sm[:, 1]
    for i in range(len(x_s) - 1):
        wi = int(round(0.5 * (x_s[i] + x_s[i + 1])))
        if wi not in index_set:
            continue
        if not (np.isfinite(y_s[i]) and np.isfinite(y_s[i + 1])):
            c = "#000000"
        else:
            c = _segment_color(win_ids, wi, meth_map, norm, cmap)
        ax.plot(
            [x_s[i] - i0, x_s[i + 1] - i0],
            [y_s[i], y_s[i + 1]],
            color=c,
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=zorder,
        )


def _plot_region_panel(
    ax,
    indices: list[int],
    win_ids: list[str],
    fp_loess: list[np.ndarray],
    both_low_loess: list[np.ndarray],
    cancer_loess: np.ndarray,
    meth_map: dict[str, float],
    norm,
    cmap,
    *,
    region_key: str,
) -> None:
    y_lim = PANEL_Y_LIMS[region_key]
    y_ref_lines = PANEL_Y_REF_LINES[region_key]
    _style_region_xaxis(ax, indices, win_ids)

    for j, sm in enumerate(fp_loess):
        _plot_loess_region(
            ax,
            sm,
            indices,
            color=GROUP_STYLE["healthy_fp"]["color"],
            zorder=8 + j,
        )
    for j, sm in enumerate(both_low_loess):
        _plot_loess_region(
            ax,
            sm,
            indices,
            color=GROUP_STYLE["healthy_both_low"]["color"],
            zorder=11 + j,
        )

    _plot_meth_colored_loess_region(
        ax,
        cancer_loess,
        indices,
        win_ids,
        meth_map,
        norm,
        cmap,
    )
    for y in y_ref_lines:
        ax.axhline(
            y,
            color="white",
            linestyle=(0, (4, 4)),
            linewidth=0.9,
            alpha=0.55,
            zorder=1,
        )
    ax.set_yticks(list(y_ref_lines))
    ax.set_ylim(*y_lim)


def _add_ellipsis_gap(fig, gs, col: int) -> None:
    gap_ax = fig.add_subplot(gs[0, col])
    gap_ax.set_axis_off()
    gap_ax.text(
        0.5,
        0.5,
        "...",
        transform=gap_ax.transAxes,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="0.35",
    )


def main() -> int:
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

    win_ids, _h_plot, c_plot, global_mean, keep = _load_training_processed()

    fp_curves: list[np.ndarray] = []
    both_low_curves: list[np.ndarray] = []
    for sample_id, _short_id in GROUP_SAMPLES["healthy_fp"]:
        fp_curves.append(_sample_curve(sample_id, keep) - global_mean)
    for sample_id, _short_id in GROUP_SAMPLES["healthy_both_low"]:
        both_low_curves.append(_sample_curve(sample_id, keep) - global_mean)

    cancer_mean = np.nanmean(c_plot, axis=0)
    fp_loess = [_precompute_loess(curve) for curve in fp_curves]
    both_low_loess = [_precompute_loess(curve) for curve in both_low_curves]
    cancer_loess = _precompute_loess(cancer_mean)
    meth_map, vmin, vmax = load_meth_mean_difference()
    from matplotlib import cm
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.RdBu_r

    region_indices = [_region_indices(win_ids, r["key"]) for r in FOCUS_REGIONS]
    grid_ratios: list[float] = []
    for i, n_win in enumerate(len(idx) for idx in region_indices):
        grid_ratios.append(n_win)
        if i < len(FOCUS_REGIONS) - 1:
            grid_ratios.append(GAP_WIDTH)

    fig = plt.figure(figsize=(13.5, 5.5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=grid_ratios, wspace=0.08)
    axes = [fig.add_subplot(gs[0, col]) for col in (0, 2)]
    _add_ellipsis_gap(fig, gs, 1)

    group_handles = [
        Line2D(
            [0],
            [0],
            color=GROUP_STYLE["healthy_fp"]["color"],
            linewidth=SAMPLE_LINE_WIDTH,
            label=GROUP_STYLE["healthy_fp"]["label"],
        ),
        Line2D(
            [0],
            [0],
            color=GROUP_STYLE["healthy_both_low"]["color"],
            linewidth=SAMPLE_LINE_WIDTH,
            label=GROUP_STYLE["healthy_both_low"]["label"],
        ),
    ]

    for ax, indices, region in zip(axes, region_indices, FOCUS_REGIONS):
        _plot_region_panel(
            ax,
            indices,
            win_ids,
            fp_loess,
            both_low_loess,
            cancer_loess,
            meth_map,
            norm,
            cmap,
            region_key=region["key"],
        )

    axes[0].set_ylabel("Short/long ratio (z-scored)", fontsize=FONT_LABEL, fontweight="bold")
    for ax in axes:
        ax.tick_params(axis="y", labelsize=FONT_TICK_Y)
        for lbl in ax.get_yticklabels():
            lbl.set_fontweight("bold")

    cancer_handle = Line2D(
        [0],
        [0],
        color=cmap(norm(np.nanmedian(list(meth_map.values())))),
        linewidth=CANCER_LINE_WIDTH,
        label="Lung cancer mean (training, meth Δβ colored)",
    )
    axes[1].legend(
        handles=group_handles + [cancer_handle],
        loc="upper right",
        framealpha=0.92,
        prop={"weight": "bold", "size": FONT_LEGEND},
    )

    fig.subplots_adjust(left=0.08, right=0.88, top=0.94, bottom=0.24)
    add_meth_colorbar(fig, norm, cmap)
    for region, indices in zip(FOCUS_REGIONS, region_indices):
        print(f"Region {region['key']}: {len(indices)} windows")
    out_png = OUT_DIR / "figure5_frag_length_top3_fp_healthy_sl_ratio_vs_cancer_mean.png"
    out_pdf = OUT_DIR / "figure5_frag_length_top3_fp_healthy_sl_ratio_vs_cancer_mean.pdf"
    for out in (out_png, out_pdf):
        fig.savefig(out, dpi=300)
        print(f"Wrote {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
