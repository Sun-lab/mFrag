#!/usr/bin/env python3
"""
Figure S3 — mFrag-Net frag-only vs frag+meth bootstrap boxplot (Testing AUC / PRAUC).
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
FD_DIR = MATHIOS / "FDConv"
OUT_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure S3"
FIG4_DIR = CFCRAFT / "mFrag_Figures" / "Figure4"

if str(FIG4_DIR) not in sys.path:
    sys.path.insert(0, str(FIG4_DIR))
from figure4_plot_titles import bootstrap_plot_title  # noqa: E402

FRAGONLY_DRAWS = FD_DIR / "fragonly_new_rep135_bootstrap_compare" / "bootstrap_draws.csv"
FRAGONLY_NPZ = FD_DIR / "fragonly_new" / "runs" / "rep_135.npz"
FRAG_METH3_DRAWS = FD_DIR / "frag_meth3ch_new_rep031_bootstrap_compare" / "bootstrap_draws.csv"

GROUPS = (
    ("mfrag_net_fragonly", "mFrag-Net (Frag length without DNA Methyl)"),
    ("mfrag_net_fragmeth", "mFrag-Net (Frag length)"),
)
GROUP_LABELS = {k: v for k, v in GROUPS}
COLORS = {
    "mfrag_net_fragonly": "#2ca02c",
    "mfrag_net_fragmeth": "#9467bd",
}

FONT_BASE = 14
FONT_TITLE = 17
FONT_LABEL = 15
FONT_TICK = 13
FONT_LEGEND = 11
FONT_BRACKET = 12
BOX_SPAN = 0.34
BOX_WIDTH = 0.14
BOX_XPAD = 0.22
REF_KEY = "mfrag_net_fragmeth"
COMP_KEY = "mfrag_net_fragonly"
BRACKET_Y = {"auc": 1.02, "prauc": 0.88}


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
    if p < 0.001:
        return f"p={p:.1e}"
    s = f"{p:.4f}".rstrip("0").rstrip(".")
    return f"p={s}"


def _mwu_fragmeth_vs_fragonly(data: dict[str, dict[str, np.ndarray]], metric: str) -> pd.Series:
    ref = data[REF_KEY][metric]
    comp = data[COMP_KEY][metric]
    res = mannwhitneyu(ref, comp, alternative="greater", method="auto")
    return pd.Series(
        {
            "metric": metric,
            "reference_key": REF_KEY,
            "reference_label": GROUP_LABELS[REF_KEY],
            "comparator_key": COMP_KEY,
            "comparator_label": GROUP_LABELS[COMP_KEY],
            "u_stat": float(res.statistic),
            "p_value": float(res.pvalue),
            "median_fragmeth": float(np.median(ref)),
            "median_fragonly": float(np.median(comp)),
            "alternative": "frag+meth > frag-only",
        }
    )


def _load_sample_ids() -> list[str]:
    z = np.load(FRAGONLY_NPZ, allow_pickle=True)
    return [str(f).replace(".csv", "") for f in z["sample_files"]]


def _load_series() -> dict[str, dict[str, np.ndarray]]:
    frag = pd.read_csv(FRAGONLY_DRAWS)
    fmeth = pd.read_csv(FRAG_METH3_DRAWS)
    return {
        "mfrag_net_fragonly": {
            "auc": frag["test_auc"].to_numpy(dtype=np.float64),
            "prauc": frag["test_prauc"].to_numpy(dtype=np.float64),
        },
        "mfrag_net_fragmeth": {
            "auc": fmeth["test_auc"].to_numpy(dtype=np.float64),
            "prauc": fmeth["test_prauc"].to_numpy(dtype=np.float64),
        },
    }


def _plot(
    *,
    out_png: Path,
    out_pdf: Path,
    ylim: tuple[float, float],
    dpi: int,
    mwu: pd.DataFrame,
    plot_title: str,
) -> None:
    data = _load_series()
    metric_keys = ("auc", "prauc")
    metric_labels = ("Testing AUC", "Testing PRAUC")
    group_centers = np.array([1.0, 2.0], dtype=float)
    n_boxes = len(GROUPS)
    offsets = np.linspace(-BOX_SPAN / 2, BOX_SPAN / 2, n_boxes)

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

    positions: list[float] = []
    box_data: list[np.ndarray] = []
    colors_order: list[str] = []
    pos_map: dict[tuple[str, str], float] = {}

    for mi, mkey in enumerate(metric_keys):
        gc = group_centers[mi]
        for gi, (gkey, _) in enumerate(GROUPS):
            pos = gc + offsets[gi]
            positions.append(pos)
            box_data.append(data[gkey][mkey])
            colors_order.append(gkey)
            pos_map[(mkey, gkey)] = pos

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    bp = ax.boxplot(
        box_data,
        positions=positions,
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
    max_pts = 60
    y_lo, y_hi = ylim
    for mkey in metric_keys:
        for gkey, _ in GROUPS:
            pos = pos_map[(mkey, gkey)]
            pts = np.asarray(data[gkey][mkey], dtype=np.float64)
            pts = pts[(pts >= y_lo) & (pts <= y_hi) & np.isfinite(pts)]
            if pts.size == 0:
                continue
            k = min(pts.size, max_pts)
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
    h = 0.018 * y_span
    text_pad = 0.010 * y_span
    line_gap = 0.038 * y_span
    for _, row in mwu.iterrows():
        mkey = str(row["metric"])
        x1 = pos_map[(mkey, COMP_KEY)]
        x2 = pos_map[(mkey, REF_KEY)]
        x_left, x_right = sorted((x1, x2))
        y_line = BRACKET_Y[mkey]
        x_mid = (x_left + x_right) / 2.0
        sig = _sig_star(float(row["p_value"]))
        p_fmt = _format_p(float(row["p_value"]))
        ax.plot(
            [x_left, x_left, x_right, x_right],
            [y_line, y_line + h, y_line + h, y_line],
            color="black",
            lw=1.4,
        )
        y_base = y_line + h + text_pad
        ax.text(
            x_mid,
            y_base + line_gap,
            rf"$\mathit{{{p_fmt}}}$",
            ha="center",
            va="bottom",
            fontsize=FONT_BRACKET,
        )
        ax.text(
            x_mid,
            y_base,
            rf"$\mathbf{{{sig}}}$",
            ha="center",
            va="bottom",
            fontsize=FONT_BRACKET,
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

    legend_handles = [
        Patch(facecolor=COLORS[k], edgecolor="black", linewidth=1.2, alpha=0.35, label=lbl)
        for k, lbl in GROUPS
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.95, edgecolor="0.75")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    data = _load_series()
    mwu = pd.DataFrame([_mwu_fragmeth_vs_fragonly(data, m) for m in ("auc", "prauc")])
    mwu["sig"] = mwu["p_value"].map(_sig_star)
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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(
        args.out_dir / "figure_s3_mfragnet_fragonly_vs_fragmeth_bootstrap_summary.csv",
        index=False,
    )
    pd.DataFrame(long_rows).to_csv(
        args.out_dir / "figure_s3_mfragnet_fragonly_vs_fragmeth_bootstrap_draws.csv",
        index=False,
    )
    mwu.to_csv(args.out_dir / "figure_s3_mfragnet_fragonly_vs_fragmeth_bootstrap_mwu.csv", index=False)
    print(mwu.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    sample_ids = _load_sample_ids()
    plot_title = bootstrap_plot_title("overall", "All stages", sample_ids)

    _plot(
        out_png=args.out_dir / "figure_s3_mfragnet_fragonly_vs_fragmeth_bootstrap_boxplot.png",
        out_pdf=args.out_dir / "figure_s3_mfragnet_fragonly_vs_fragmeth_bootstrap_boxplot.pdf",
        ylim=(0.0, 1.2),
        dpi=args.dpi,
        mwu=mwu,
        plot_title=plot_title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
