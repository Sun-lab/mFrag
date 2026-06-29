#!/usr/bin/env python3
"""
Figure S4 — four-model DELFI / mFrag-DELFI bootstrap boxplot (Testing AUC / PRAUC).

Models (left to right):
  1. DELFI (S/L ratio)           — pca66 Lasso baseline
  2. DELFI (Concatenation)       — concat S/L+Shannon Lasso baseline
  3. mFrag-DELFI (Frag length)   — pca66 Lasso |Δβ|/mean coupling
  4. mFrag-DELFI (Concatenation) — concat S/L+Shannon Lasso coupling
"""
from __future__ import annotations

import argparse
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
SL_DIR = CFCRAFT / "Mathios" / "Mathios_slratio_method"
PCA66_BOOT = SL_DIR / "pca66_permuted_coupling_bootstrap_compare" / "bootstrap_draws.csv"
CONCAT_BOOT = SL_DIR / "concat_slratio_shannon_bootstrap_compare" / "bootstrap_draws.csv"
OUT_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure S4"

GROUPS = (
    ("delfi_sl_ratio", "DELFI (S/L ratio)"),
    ("delfi_concatenation", "DELFI (S/L ratio + MDS)"),
    ("mfrag_delfi_frag_length", "mFrag-Weight (S/L ratio)"),
    ("mfrag_delfi_concatenation", "mFrag-Weight (S/L ratio + MDS)"),
)
GROUP_LABELS = {k: v for k, v in GROUPS}
SOURCE = {
    "delfi_sl_ratio": ("pca66", "baseline"),
    "delfi_concatenation": ("concat", "baseline"),
    "mfrag_delfi_frag_length": ("pca66", "mean"),
    "mfrag_delfi_concatenation": ("concat", "coupling"),
}
COLORS = {
    "delfi_sl_ratio": "#1f77b4",
    "delfi_concatenation": "#17becf",
    "mfrag_delfi_frag_length": "#ff7f0e",
    "mfrag_delfi_concatenation": "#2ca02c",
}

FONT_BASE = 14
FONT_TITLE = 17
FONT_LABEL = 15
FONT_TICK = 13
FONT_LEGEND = 10
BOX_SPAN = 0.58
BOX_WIDTH = 0.12
BOX_XPAD = 0.30
PLOT_TITLE = "All stages (testing healthy = 385, testing cancer = 46)"
# (baseline, concatenation) — one-sided MWU: concatenation > baseline
PAIR_COMPARISONS = (
    ("delfi_sl_ratio", "delfi_concatenation"),
    ("mfrag_delfi_frag_length", "mfrag_delfi_concatenation"),
)
BRACKET_Y = {"auc": 1.05, "prauc": 0.77}
FONT_BRACKET = 11


def _load_lasso_draws(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    lasso = df[df["family"] == "Lasso"].copy()
    if lasso.empty:
        raise SystemExit(f"No Lasso rows in {csv_path}")
    return lasso


def _load_series() -> dict[str, dict[str, np.ndarray]]:
    pca66 = _load_lasso_draws(PCA66_BOOT)
    concat = _load_lasso_draws(CONCAT_BOOT)
    out: dict[str, dict[str, np.ndarray]] = {}
    for gkey, (src, variant) in SOURCE.items():
        sub = pca66 if src == "pca66" else concat
        rows = sub[sub["variant"] == variant]
        if rows.empty:
            raise SystemExit(f"Missing {src} Lasso variant={variant} for {gkey}")
        out[gkey] = {
            "auc": rows["test_auc"].to_numpy(dtype=np.float64),
            "prauc": rows["test_prauc"].to_numpy(dtype=np.float64),
        }
    return out


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


def _mwu_pairs(data: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for metric in ("auc", "prauc"):
        for baseline_key, concat_key in PAIR_COMPARISONS:
            baseline = data[baseline_key][metric]
            concat = data[concat_key][metric]
            res = mannwhitneyu(concat, baseline, alternative="greater", method="auto")
            rows.append(
                {
                    "metric": metric,
                    "baseline_key": baseline_key,
                    "baseline_label": GROUP_LABELS[baseline_key],
                    "concat_key": concat_key,
                    "concat_label": GROUP_LABELS[concat_key],
                    "u_stat": float(res.statistic),
                    "p_value": float(res.pvalue),
                    "median_baseline": float(np.median(baseline)),
                    "median_concat": float(np.median(concat)),
                    "alternative": "concatenation > baseline",
                    "sig": _sig_star(float(res.pvalue)),
                }
            )
    return pd.DataFrame(rows)


def _plot(
    *,
    data: dict[str, dict[str, np.ndarray]],
    out_png: Path,
    out_pdf: Path,
    ylim: tuple[float, float],
    dpi: int,
    mwu: pd.DataFrame,
) -> None:
    metric_keys = ("auc", "prauc")
    metric_labels = ("Testing AUC", "Testing PRAUC")
    group_centers = np.array([1.0, 2.0], dtype=float)
    offsets = np.linspace(-BOX_SPAN / 2, BOX_SPAN / 2, len(GROUPS))

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

    fig, ax = plt.subplots(figsize=(11.0, 6.8))
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
    for mi, mkey in enumerate(metric_keys):
        gc = group_centers[mi]
        for gi, (gkey, _) in enumerate(GROUPS):
            pos = gc + offsets[gi]
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
        left_key = str(row["baseline_key"])
        right_key = str(row["concat_key"])
        x1 = pos_map[(mkey, left_key)]
        x2 = pos_map[(mkey, right_key)]
        x_left, x_right = sorted((x1, x2))
        y_line = BRACKET_Y[mkey]
        x_mid = (x_left + x_right) / 2.0
        sig = str(row["sig"])
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
    ax.set_title(PLOT_TITLE, fontsize=FONT_TITLE, fontweight="bold")
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
    ap.add_argument("--ylim", type=float, nargs=2, default=(0.0, 1.4))
    args = ap.parse_args()

    data = _load_series()
    mwu = _mwu_pairs(data)
    summary_rows = []
    long_rows = []
    for gkey, glabel in GROUPS:
        src, variant = SOURCE[gkey]
        for mkey in ("auc", "prauc"):
            arr = data[gkey][mkey]
            summary_rows.append(
                {
                    "group_key": gkey,
                    "group_label": glabel,
                    "source": src,
                    "variant": variant,
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
                        "source": src,
                        "variant": variant,
                        "metric": mkey,
                        "bootstrap_idx": i,
                        "value": float(v),
                    }
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(
        args.out_dir / "figure_s4_delfi_mfragdelfi_four_model_bootstrap_summary.csv",
        index=False,
    )
    pd.DataFrame(long_rows).to_csv(
        args.out_dir / "figure_s4_delfi_mfragdelfi_four_model_bootstrap_draws.csv",
        index=False,
    )
    mwu.to_csv(args.out_dir / "figure_s4_delfi_mfragdelfi_four_model_bootstrap_mwu.csv", index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(mwu.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    _plot(
        data=data,
        out_png=args.out_dir / "figure_s4_delfi_mfragdelfi_four_model_bootstrap_boxplot.png",
        out_pdf=args.out_dir / "figure_s4_delfi_mfragdelfi_four_model_bootstrap_boxplot.pdf",
        ylim=tuple(args.ylim),
        dpi=args.dpi,
        mwu=mwu,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
