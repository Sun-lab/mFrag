#!/usr/bin/env python3
"""
Figure S5 — DELFI (S/L ratio) PCA+L1 vs DELFI (S/L ratio) L1 without PCA.

Bootstrap boxplot (Testing AUC / Testing PRAUC).
Data: Mathios/Mathios_slratio_method/pca_vs_nopca_bootstrap_compare/bootstrap_draws.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu

CFCRAFT = Path("/fh/working/sun_w/saishi_cui/cfCraft")
MATHIOS = CFCRAFT / "Mathios"
SL_DIR = MATHIOS / "Mathios_slratio_method"
FIG4_DIR = CFCRAFT / "mFrag_Figures" / "Figure4"
OUT_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure S5"

if str(FIG4_DIR) not in sys.path:
    sys.path.insert(0, str(FIG4_DIR))
from figure4_plot_titles import bootstrap_plot_title  # noqa: E402

BOOT_DRAWS = SL_DIR / "pca_vs_nopca_bootstrap_compare" / "bootstrap_draws.csv"
SL_TEST_PRED = SL_DIR / "pca66_l1_baseline" / "test_predictions.csv"

GROUPS = (
    ("pca", "DELFI (S/L ratio)"),
    ("nopca", "DELFI (S/L ratio) (without PCA)"),
)
GROUP_LABELS = {k: v for k, v in GROUPS}
COLORS = {"pca": "#1f77b4", "nopca": "#ff7f0e"}

FONT_BASE = 14
FONT_TITLE = 17
FONT_LABEL = 15
FONT_TICK = 13
FONT_LEGEND = 10
FONT_BRACKET = 10
BOX_SPAN = 0.34
BOX_WIDTH = 0.14
BOX_XPAD = 0.22


def _sig_star(p: float) -> str:
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def _format_p(p: float) -> str:
    if not np.isfinite(p) or p <= 0.0:
        return "p<1e-300"
    if p < 0.001:
        return f"p={p:.1e}"
    s = f"{p:.4f}".rstrip("0").rstrip(".")
    return f"p={s}"


def _load_sample_ids() -> list[str]:
    df = pd.read_csv(SL_TEST_PRED)
    return df["sample_id"].astype(str).tolist()


def _load_series() -> dict[str, dict[str, np.ndarray]]:
    draws = pd.read_csv(BOOT_DRAWS)
    data: dict[str, dict[str, np.ndarray]] = {}
    for gkey in ("pca", "nopca"):
        sub = draws[draws["model"] == gkey]
        data[gkey] = {
            "auc": sub["test_auc"].to_numpy(dtype=np.float64),
            "prauc": sub["test_prauc"].to_numpy(dtype=np.float64),
        }
    return data


def _mwu_pca_vs_nopca(data: dict[str, dict[str, np.ndarray]], metric: str) -> pd.DataFrame:
    pca = data["pca"][metric]
    nopca = data["nopca"][metric]
    res = mannwhitneyu(pca, nopca, alternative="two-sided", method="auto")
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "group_a_key": "pca",
                "group_a_label": GROUP_LABELS["pca"],
                "group_b_key": "nopca",
                "group_b_label": GROUP_LABELS["nopca"],
                "u_stat": float(res.statistic),
                "p_value": float(res.pvalue),
                "median_pca": float(np.median(pca)),
                "median_nopca": float(np.median(nopca)),
            }
        ]
    )


def _plot_boxplot(
    *,
    data: dict[str, dict[str, np.ndarray]],
    out_png: Path,
    ylim: tuple[float, float],
    dpi: int,
    mwu_auc: pd.DataFrame,
    mwu_pr: pd.DataFrame,
    plot_title: str,
) -> None:
    metric_keys = ("auc", "prauc")
    metric_labels = ("Testing AUC", "Testing PRAUC")
    group_centers = np.array([1.0, 2.0], dtype=float)
    offsets = np.array([-BOX_SPAN / 2, BOX_SPAN / 2])
    pos_map: dict[tuple[str, str], float] = {}

    mpl.rcParams.update(
        {
            "font.size": FONT_BASE,
            "axes.titlesize": FONT_TITLE,
            "axes.labelsize": FONT_LABEL,
            "legend.fontsize": FONT_LEGEND,
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.linewidth": 1.2,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    box_data: list[np.ndarray] = []
    colors_order: list[str] = []
    for mi, mkey in enumerate(metric_keys):
        for gi, (gkey, _) in enumerate(GROUPS):
            pos = group_centers[mi] + offsets[gi]
            pos_map[(mkey, gkey)] = pos
            box_data.append(data[gkey][mkey])
            colors_order.append(gkey)

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    bp = ax.boxplot(
        box_data,
        positions=[pos_map[(mk, gk)] for mk in metric_keys for gk, _ in GROUPS],
        widths=BOX_WIDTH,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 2.2},
        whiskerprops={"color": "black", "linewidth": 1.5},
        capprops={"color": "black", "linewidth": 1.5},
        boxprops={"edgecolor": "black", "linewidth": 1.5},
    )
    for i, gkey in enumerate(colors_order):
        bp["boxes"][i].set_facecolor(COLORS[gkey])
        bp["boxes"][i].set_alpha(0.35)

    rng = np.random.default_rng(7)
    y_lo, y_hi = ylim
    for mkey in metric_keys:
        for gkey, _ in GROUPS:
            pos = pos_map[(mkey, gkey)]
            pts = np.asarray(data[gkey][mkey], dtype=np.float64)
            pts = pts[(pts >= y_lo) & (pts <= y_hi) & np.isfinite(pts)]
            if pts.size == 0:
                continue
            k = min(pts.size, 60)
            if k < pts.size:
                pts = pts[rng.choice(pts.size, size=k, replace=False)]
            base = COLORS[gkey]
            r, g, b = to_rgb(base) if mcolors.is_color_like(base) else (0.4, 0.4, 0.4)
            light = (0.55 + 0.45 * r, 0.55 + 0.45 * g, 0.55 + 0.45 * b)
            jitter = rng.uniform(-0.04, 0.04, size=pts.size)
            ax.scatter(
                np.full_like(pts, pos) + jitter,
                pts,
                s=14,
                color=light,
                alpha=0.55,
                linewidths=0.0,
                zorder=3,
            )

    y_span = ylim[1] - ylim[0]
    h = 0.010 * y_span
    text_pad = 0.007 * y_span
    line_gap = 0.030 * y_span
    for mkey, mwu_df in (("auc", mwu_auc), ("prauc", mwu_pr)):
        row = mwu_df.iloc[0]
        x1 = pos_map[(mkey, "pca")]
        x2 = pos_map[(mkey, "nopca")]
        x_left, x_right = sorted((x1, x2))
        panel_vals = np.concatenate([data[gkey][mkey] for gkey, _ in GROUPS])
        data_top = float(np.max(panel_vals[np.isfinite(panel_vals)]))
        y_line = data_top + 0.04 * y_span
        x_mid = (x_left + x_right) / 2.0
        sig = _sig_star(float(row["p_value"]))
        p_fmt = _format_p(float(row["p_value"]))
        ax.plot(
            [x_left, x_left, x_right, x_right],
            [y_line, y_line + h, y_line + h, y_line],
            color="black",
            lw=1.4,
            clip_on=False,
        )
        y_stars = y_line + h + text_pad
        y_p = y_stars + line_gap
        ax.text(
            x_mid,
            y_stars,
            rf"$\mathbf{{{sig}}}$",
            ha="center",
            va="bottom",
            fontsize=FONT_BRACKET,
            clip_on=False,
        )
        ax.text(
            x_mid,
            y_p,
            rf"$\mathit{{{p_fmt}}}$",
            ha="center",
            va="bottom",
            fontsize=FONT_BRACKET,
            clip_on=False,
        )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(metric_labels, fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title(plot_title, fontsize=FONT_TITLE, fontweight="bold")
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="both", which="both", width=1.2, labelsize=FONT_TICK)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

    left = group_centers[0] + offsets[0] - BOX_WIDTH / 2 - BOX_XPAD
    right = group_centers[1] + offsets[-1] + BOX_WIDTH / 2 + BOX_XPAD
    ax.set_xlim(left, right)

    ax.legend(
        handles=[
            Patch(facecolor=COLORS[k], edgecolor="black", linewidth=1.2, alpha=0.35, label=lbl)
            for k, lbl in GROUPS
        ],
        loc="upper right",
        framealpha=0.95,
        edgecolor="0.75",
        prop={"size": FONT_LEGEND, "weight": "bold"},
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    if not BOOT_DRAWS.is_file():
        raise SystemExit(f"Missing {BOOT_DRAWS} — run slratio_pca_vs_nopca_bootstrap_boxplot_compare.py first.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = _load_series()
    mwu_auc = _mwu_pca_vs_nopca(data, "auc")
    mwu_pr = _mwu_pca_vs_nopca(data, "prauc")
    mwu = pd.concat([mwu_auc, mwu_pr], ignore_index=True)
    mwu.to_csv(args.out_dir / "figure_s5_delfi_slratio_pca_vs_nopca_bootstrap_mwu.csv", index=False)
    print(mwu.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    summary_rows = []
    long_rows = []
    for gkey, glabel in GROUPS:
        for mkey in ("auc", "prauc"):
            arr = data[gkey][mkey]
            summary_rows.append(
                {
                    "group_key": gkey,
                    "group_label": glabel,
                    "metric": mkey,
                    "n": len(arr),
                    "median": float(np.median(arr)),
                    "q25": float(np.percentile(arr, 25)),
                    "q75": float(np.percentile(arr, 75)),
                }
            )
            for i, v in enumerate(arr):
                long_rows.append(
                    {
                        "group_key": gkey,
                        "group_label": glabel,
                        "metric": mkey,
                        "bootstrap_idx": i,
                        "value": float(v),
                    }
                )
    pd.DataFrame(summary_rows).to_csv(
        args.out_dir / "figure_s5_delfi_slratio_pca_vs_nopca_bootstrap_summary.csv",
        index=False,
    )
    pd.DataFrame(long_rows).to_csv(
        args.out_dir / "figure_s5_delfi_slratio_pca_vs_nopca_bootstrap_draws.csv",
        index=False,
    )

    plot_title = bootstrap_plot_title("overall", "All stages", _load_sample_ids())
    _plot_boxplot(
        data=data,
        out_png=args.out_dir / "figure_s5_delfi_slratio_pca_vs_nopca_bootstrap_boxplot.png",
        ylim=(0.0, 1.15),
        dpi=args.dpi,
        mwu_auc=mwu_auc,
        mwu_pr=mwu_pr,
        plot_title=plot_title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
