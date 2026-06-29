#!/usr/bin/env python3
"""
Figure 4 — three-model bootstrap boxplot for End 4-mer motif (Testing AUC / PRAUC).

Models:
  1. DELFI (MDS)                 — Shannon PCA54 Lasso baseline
  2. mFrag-Weight (MDS)            — Shannon PCA54 Lasso 1−λ×Δβ̃ coupling
  3. mFrag-Net (Motif)  — FDConv motif_meth3ch rep006
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

from figure4_plot_titles import bootstrap_plot_title

CFCRAFT = Path("/fh/working/sun_w/saishi_cui/cfCraft")
MATHIOS = CFCRAFT / "Mathios"
OUT_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure4"

SHANNON_TEST_PRED = MATHIOS / "Mathios_slratio_method" / "shannon_pca54_l1_baseline" / "test_predictions.csv"
SHANNON_DRAWS = MATHIOS / "Mathios_slratio_method" / "shannon_pca54_bootstrap_compare" / "bootstrap_draws.csv"
MOTIF_METH3_DRAWS = MATHIOS / "FDConv" / "motif_meth3ch_new_rep006_bootstrap_compare" / "bootstrap_draws.csv"

GROUPS = (
    ("delfi", "DELFI (MDS)"),
    ("mfrag_delfi", "mFrag-Weight (MDS)"),
    ("mfrag_net_ref", "mFrag-Net (Motif)"),
)
GROUP_LABELS = {k: v for k, v in GROUPS}
COLORS = {
    "delfi": "#1f77b4",
    "mfrag_delfi": "#ff7f0e",
    "mfrag_net_ref": "#9467bd",
}
REF_KEY = "delfi"
COMP_KEYS = ("mfrag_delfi", "mfrag_net_ref")
BRACKET_Y = {"auc": (1.02, 1.18), "prauc": (0.88, 1.06)}
FONT_BASE = 14
FONT_TITLE = 17
FONT_LABEL = 15
FONT_TICK = 13
FONT_LEGEND = 12
FONT_BRACKET = 12
BOX_SPAN = 0.46
BOX_WIDTH = 0.14
BOX_XPAD = 0.20


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


def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64)
    m = int(p.size)
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(m, dtype=np.float64)
    prev = 0.0
    for i, pv in enumerate(ranked):
        k = m - i
        val = min(1.0, pv * k)
        val = max(val, prev)
        adj[i] = val
        prev = val
    out = np.empty(m, dtype=np.float64)
    out[order] = adj
    return out


def _mwu_delfi_vs_others(
    data: dict[str, dict[str, np.ndarray]],
    metric: str,
) -> pd.DataFrame:
    ref = data[REF_KEY][metric]
    rows = []
    for comp in COMP_KEYS:
        other = data[comp][metric]
        res = mannwhitneyu(other, ref, alternative="greater", method="auto")
        rows.append(
            {
                "metric": metric,
                "reference_key": REF_KEY,
                "reference_label": GROUP_LABELS[REF_KEY],
                "comparator_key": comp,
                "comparator_label": GROUP_LABELS[comp],
                "u_stat": float(res.statistic),
                "p_raw": float(res.pvalue),
                "median_delfi": float(np.median(ref)),
                "median_comparator": float(np.median(other)),
                "alternative": "comparator > DELFI",
            }
        )
    df = pd.DataFrame(rows)
    df["p_holm"] = _holm_adjust(df["p_raw"].to_numpy())
    df["sig_holm"] = df["p_holm"].map(_sig_star)
    return df


def _load_test_sample_ids() -> list[str]:
    df = pd.read_csv(SHANNON_TEST_PRED)
    return df["sample_id"].astype(str).tolist()


def _load_series() -> dict[str, dict[str, np.ndarray]]:
    sh = pd.read_csv(SHANNON_DRAWS)
    lasso_base = sh[(sh["family"] == "Lasso") & (sh["variant"] == "baseline")]
    lasso_coupling = sh[(sh["family"] == "Lasso") & (sh["variant"] == "one_minus_lambda")]
    mmeth = pd.read_csv(MOTIF_METH3_DRAWS)
    return {
        "delfi": {
            "auc": lasso_base["test_auc"].to_numpy(dtype=np.float64),
            "prauc": lasso_base["test_prauc"].to_numpy(dtype=np.float64),
        },
        "mfrag_delfi": {
            "auc": lasso_coupling["test_auc"].to_numpy(dtype=np.float64),
            "prauc": lasso_coupling["test_prauc"].to_numpy(dtype=np.float64),
        },
        "mfrag_net_ref": {
            "auc": mmeth["test_auc"].to_numpy(dtype=np.float64),
            "prauc": mmeth["test_prauc"].to_numpy(dtype=np.float64),
        },
    }


def _plot(
    *,
    out_png: Path,
    out_pdf: Path,
    ylim: tuple[float, float],
    dpi: int,
    mwu_auc: pd.DataFrame,
    mwu_pr: pd.DataFrame,
    plot_title: str,
) -> None:
    data = _load_series()
    metric_keys = ("auc", "prauc")
    metric_labels = ("Testing AUC", "Testing PRAUC")
    group_centers = np.array([1.0, 2.0], dtype=float)
    n_boxes = len(GROUPS)
    span = BOX_SPAN
    offsets = np.linspace(-span / 2, span / 2, n_boxes)
    width = BOX_WIDTH

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

    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=width,
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

    for mkey, mwu_df in (("auc", mwu_auc), ("prauc", mwu_pr)):
        levels = BRACKET_Y[mkey]
        x_ref = pos_map[(mkey, REF_KEY)]
        for bi, row in mwu_df.reset_index(drop=True).iterrows():
            x_comp = pos_map[(mkey, row["comparator_key"])]
            x1, x2 = sorted((x_comp, x_ref))
            y_line = levels[bi] if bi < len(levels) else levels[-1]
            x_mid = (x1 + x2) / 2.0
            sig = str(row["sig_holm"])
            p_fmt = _format_p(float(row["p_holm"]))
            ax.plot([x1, x1, x2, x2], [y_line, y_line + h, y_line + h, y_line], color="black", lw=1.4)
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

    left = group_centers[0] + offsets[0] - width / 2 - BOX_XPAD
    right = group_centers[1] + offsets[-1] + width / 2 + BOX_XPAD
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
    sample_ids = _load_test_sample_ids()
    plot_title = bootstrap_plot_title("overall", "All stages", sample_ids)
    mwu_auc = _mwu_delfi_vs_others(data, "auc")
    mwu_pr = _mwu_delfi_vs_others(data, "prauc")
    mwu = pd.concat([mwu_auc, mwu_pr], ignore_index=True)

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
    pd.DataFrame(summary_rows).to_csv(args.out_dir / "figure4_end4motif_bootstrap_summary.csv", index=False)
    pd.DataFrame(long_rows).to_csv(args.out_dir / "figure4_end4motif_bootstrap_draws.csv", index=False)
    mwu.to_csv(args.out_dir / "figure4_end4motif_mwu_delfi_vs_others.csv", index=False)
    print(mwu.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    _plot(
        out_png=args.out_dir / "figure4_end4motif_bootstrap_boxplot.png",
        out_pdf=args.out_dir / "figure4_end4motif_bootstrap_boxplot.pdf",
        ylim=(0.0, 1.55),
        dpi=args.dpi,
        mwu_auc=mwu_auc,
        mwu_pr=mwu_pr,
        plot_title=plot_title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
