#!/usr/bin/env python3
"""
Figure S5 — three-model mFrag-Net bootstrap boxplot + ROC / PR-ROC.

Models (left to right):
  1. mFrag-Net (Frag length)
  2. mFrag-Net (End 4-mer motif)
  3. mFrag-Net (Fusion)

Data:
  frag_meth3ch_new_rep031_bootstrap_compare
  motif_meth3ch_new_rep006_bootstrap_compare
  frag_motif_meth4ch_new_rep31_bootstrap_compare
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
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

CFCRAFT = Path("/fh/working/sun_w/saishi_cui/cfCraft")
MATHIOS = CFCRAFT / "Mathios"
FD_DIR = MATHIOS / "FDConv"
MATHIOS_SCRIPTS = MATHIOS / "scripts"
FIG4_DIR = CFCRAFT / "mFrag_Figures" / "Figure4"
OUT_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure S5"

if str(MATHIOS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MATHIOS_SCRIPTS))
if str(FIG4_DIR) not in sys.path:
    sys.path.insert(0, str(FIG4_DIR))
from figure4_plot_titles import bootstrap_plot_title  # noqa: E402
from roc_plot_utils import pad_median_iqr_for_origin_vertical, roc_median_iqr  # noqa: E402

FRAG_METH_DRAWS = FD_DIR / "frag_meth3ch_new_rep031_bootstrap_compare" / "bootstrap_draws.csv"
MOTIF_DRAWS = FD_DIR / "motif_meth3ch_new_rep006_bootstrap_compare" / "bootstrap_draws.csv"
FOURCH_DRAWS = FD_DIR / "frag_motif_meth4ch_new_rep31_bootstrap_compare" / "bootstrap_draws.csv"

FRAG_METH_NPZ = FD_DIR / "frag_meth3ch_new" / "runs" / "rep_031.npz"
MOTIF_NPZ = FD_DIR / "motif_meth3ch_new" / "runs" / "rep_006.npz"
FOURCH_NPZ = FD_DIR / "frag_motif_meth4ch_new" / "runs" / "rep_031.npz"

GROUPS = (
    ("mfrag_net_fraglength", "mFrag-Net (Frag length)"),
    ("mfrag_net_end4motif", "mFrag-Net (End 4-mer motif)"),
    ("mfrag_net_fusion", "mFrag-Net (Fusion)"),
)
GROUP_LABELS = {k: v for k, v in GROUPS}
COLORS = {
    "mfrag_net_fraglength": "#9467bd",
    "mfrag_net_end4motif": "#ff7f0e",
    "mfrag_net_fusion": "#2ca02c",
}
# Compare model 1 vs 3 and model 2 vs 3 only (each vs Fusion).
PAIR_KEYS = (
    ("mfrag_net_fraglength", "mfrag_net_fusion"),
    ("mfrag_net_end4motif", "mfrag_net_fusion"),
)

FONT_BASE = 14
FONT_TITLE = 17
FONT_LABEL = 15
FONT_TICK = 13
FONT_LEGEND = 10
FONT_BRACKET = 11
BOX_SPAN = 0.58
BOX_WIDTH = 0.12
BOX_XPAD = 0.30
PLOT_TITLE = bootstrap_plot_title("overall", "All stages", ["_healthy"] * 385 + ["_lung_cancer"] * 46)
# Vertical space reserved per bracket tier (bracket + stars + p-value).
BRACKET_BLOCK_FRAC = 0.112
BRACKET_TIER_GAP_FRAC = 0.028
BRACKET_TOP_MARGIN_FRAC = 0.036
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


def _load_series() -> dict[str, dict[str, np.ndarray]]:
    frag = pd.read_csv(FRAG_METH_DRAWS)
    motif = pd.read_csv(MOTIF_DRAWS)
    four = pd.read_csv(FOURCH_DRAWS)
    return {
        "mfrag_net_fraglength": {
            "auc": frag["test_auc"].to_numpy(dtype=np.float64),
            "prauc": frag["test_prauc"].to_numpy(dtype=np.float64),
        },
        "mfrag_net_end4motif": {
            "auc": motif["test_auc"].to_numpy(dtype=np.float64),
            "prauc": motif["test_prauc"].to_numpy(dtype=np.float64),
        },
        "mfrag_net_fusion": {
            "auc": four["test_auc"].to_numpy(dtype=np.float64),
            "prauc": four["test_prauc"].to_numpy(dtype=np.float64),
        },
    }


def _sorted_mwu_for_brackets(
    mwu_df: pd.DataFrame,
    pos_map: dict[tuple[str, str], float],
    metric: str,
) -> pd.DataFrame:
    """Wider x-span comparisons sit lower; narrower comparisons stack above."""
    rows = []
    for _, row in mwu_df.iterrows():
        x1 = pos_map[(metric, row["group_a_key"])]
        x2 = pos_map[(metric, row["group_b_key"])]
        rows.append({**row.to_dict(), "_x_span": abs(x2 - x1)})
    out = pd.DataFrame(rows)
    return out.sort_values("_x_span", ascending=False).reset_index(drop=True)


def _bracket_y_levels(
    *,
    n_brackets: int,
    data_top: float,
    y_span: float,
) -> list[float]:
    block = BRACKET_BLOCK_FRAC * y_span
    gap = BRACKET_TIER_GAP_FRAC * y_span
    margin = BRACKET_TOP_MARGIN_FRAC * y_span
    step = block + gap
    y0 = max(data_top + margin, data_top + 0.06 * y_span)
    return [y0 + i * step for i in range(n_brackets)]


def _mwu_pairwise(data: dict[str, dict[str, np.ndarray]], metric: str) -> pd.DataFrame:
    rows = []
    for a, b in PAIR_KEYS:
        x = data[a][metric]
        y = data[b][metric]
        res = mannwhitneyu(x, y, alternative="two-sided", method="auto")
        rows.append(
            {
                "metric": metric,
                "group_a_key": a,
                "group_a_label": GROUP_LABELS[a],
                "group_b_key": b,
                "group_b_label": GROUP_LABELS[b],
                "u_stat": float(res.statistic),
                "p_raw": float(res.pvalue),
                "median_a": float(np.median(x)),
                "median_b": float(np.median(y)),
            }
        )
    df = pd.DataFrame(rows)
    df["p_holm"] = _holm_adjust(df["p_raw"].to_numpy())
    df["sig_holm"] = df["p_holm"].map(_sig_star)
    return df


def _plot_boxplot(
    *,
    data: dict[str, dict[str, np.ndarray]],
    out_png: Path,
    out_pdf: Path,
    ylim: tuple[float, float],
    dpi: int,
    mwu_auc: pd.DataFrame,
    mwu_pr: pd.DataFrame,
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
    h = 0.014 * y_span
    text_pad = 0.012 * y_span
    line_gap = 0.045 * y_span
    for mkey, mwu_df in (("auc", mwu_auc), ("prauc", mwu_pr)):
        panel_vals = np.concatenate([data[gkey][mkey] for gkey, _ in GROUPS])
        data_top = float(np.max(panel_vals[np.isfinite(panel_vals)]))
        ordered = _sorted_mwu_for_brackets(mwu_df, pos_map, mkey)
        levels = _bracket_y_levels(
            n_brackets=len(ordered),
            data_top=data_top,
            y_span=y_span,
        )
        for bi, row in ordered.iterrows():
            x1 = pos_map[(mkey, row["group_a_key"])]
            x2 = pos_map[(mkey, row["group_b_key"])]
            x_left, x_right = sorted((x1, x2))
            y_line = levels[bi]
            x_mid = (x_left + x_right) / 2.0
            sig = str(row["sig_holm"])
            p_fmt = _format_p(float(row["p_holm"]))
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


def _load_cnn_pred(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    return np.asarray(z["labels"], dtype=np.int32), np.asarray(z["probs"], dtype=np.float64)


def _bootstrap_curves(
    y: np.ndarray,
    prob: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[list, list, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(y)
    roc_curves, pr_curves = [], []
    aucs, praucs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b, p_b = y[idx], prob[idx]
        if len(np.unique(y_b)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_b, p_b)
        prec, rec, _ = precision_recall_curve(y_b, p_b)
        roc_curves.append((fpr, tpr))
        pr_curves.append((prec, rec))
        aucs.append(float(roc_auc_score(y_b, p_b)))
        praucs.append(float(average_precision_score(y_b, p_b)))
    return roc_curves, pr_curves, np.asarray(aucs), np.asarray(praucs)


def _pr_median_iqr(
    curves: list[tuple[np.ndarray, np.ndarray]],
    recall_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mats = []
    for prec, rec in curves:
        order = np.argsort(rec)
        mats.append(np.interp(recall_grid, rec[order], prec[order], left=1.0, right=0.0))
    arr = np.stack(mats, axis=0)
    return (
        np.median(arr, axis=0),
        np.percentile(arr, 25.0, axis=0),
        np.percentile(arr, 75.0, axis=0),
    )


def _plot_roc(
    bundles: list[tuple],
    *,
    out_png: Path,
    out_pdf: Path,
    out_csv: Path,
    fpr_points: int,
    dpi: int,
) -> None:
    fpr_grid = np.linspace(0.0, 1.0, int(fpr_points), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.35, zorder=0)
    rows = []
    for mkey, label, roc_curves, aucs in bundles:
        med, lo, hi = roc_median_iqr(roc_curves, fpr_grid)
        fpr_plot, med, lo, hi = pad_median_iqr_for_origin_vertical(fpr_grid, med, lo, hi)
        auc_med = float(np.median(aucs))
        auc_q1 = float(np.percentile(aucs, 25.0))
        auc_q3 = float(np.percentile(aucs, 75.0))
        color = COLORS[mkey]
        ax.fill_between(fpr_plot, lo, hi, color=color, alpha=0.18, linewidth=0.0, zorder=1)
        ax.plot(
            fpr_plot,
            med,
            color=color,
            lw=2.4,
            label=f"{label} ({auc_med:.3f} [{auc_q1:.3f}, {auc_q3:.3f}])",
            zorder=2,
        )
        rows.append(
            {
                "model_key": mkey,
                "label": label,
                "n_bootstrap": int(aucs.size),
                "auc_median": auc_med,
                "auc_q25": auc_q1,
                "auc_q75": auc_q3,
            }
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False positive rate", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("True positive rate", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(PLOT_TITLE, fontsize=FONT_TITLE, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, prop={"size": FONT_LEGEND, "weight": "bold"})
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_png}")


def _plot_pr(
    bundles: list[tuple],
    prevalence: float,
    *,
    out_png: Path,
    out_pdf: Path,
    out_csv: Path,
    recall_points: int,
    dpi: int,
) -> None:
    recall_grid = np.linspace(0.0, 1.0, int(recall_points), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    rows = []
    for mkey, label, pr_curves, praucs, _prev in bundles:
        med, lo, hi = _pr_median_iqr(pr_curves, recall_grid)
        pr_med = float(np.median(praucs))
        pr_q1 = float(np.percentile(praucs, 25.0))
        pr_q3 = float(np.percentile(praucs, 75.0))
        color = COLORS[mkey]
        ax.fill_between(recall_grid, lo, hi, color=color, alpha=0.18, linewidth=0.0, zorder=1)
        ax.plot(
            recall_grid,
            med,
            color=color,
            lw=2.4,
            label=f"{label} ({pr_med:.3f} [{pr_q1:.3f}, {pr_q3:.3f}])",
            zorder=2,
        )
        rows.append(
            {
                "model_key": mkey,
                "label": label,
                "n_bootstrap": int(praucs.size),
                "prauc_median": pr_med,
                "prauc_q25": pr_q1,
                "prauc_q75": pr_q3,
            }
        )
    ax.axhline(prevalence, color="k", ls="--", lw=1, alpha=0.4, label=f"prevalence = {prevalence:.3f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(PLOT_TITLE, fontsize=FONT_TITLE, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.legend(loc="upper right", framealpha=0.92, prop={"size": FONT_LEGEND, "weight": "bold"})
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fpr-points", type=int, default=2001)
    ap.add_argument("--recall-points", type=int, default=2001)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--ylim", type=float, nargs=2, default=(0.0, 1.5))
    args = ap.parse_args()

    data = _load_series()
    mwu_auc = _mwu_pairwise(data, "auc")
    mwu_pr = _mwu_pairwise(data, "prauc")
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
    pd.DataFrame(summary_rows).to_csv(
        args.out_dir / "figure_s5_mfragnet_three_model_bootstrap_summary.csv",
        index=False,
    )
    pd.DataFrame(long_rows).to_csv(
        args.out_dir / "figure_s5_mfragnet_three_model_bootstrap_draws.csv",
        index=False,
    )
    mwu.to_csv(args.out_dir / "figure_s5_mfragnet_three_model_bootstrap_mwu_holm.csv", index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(mwu.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    _plot_boxplot(
        data=data,
        out_png=args.out_dir / "figure_s5_mfragnet_three_model_bootstrap_boxplot.png",
        out_pdf=args.out_dir / "figure_s5_mfragnet_three_model_bootstrap_boxplot.pdf",
        ylim=tuple(args.ylim),
        dpi=args.dpi,
        mwu_auc=mwu_auc,
        mwu_pr=mwu_pr,
    )

    roc_bundles: list[tuple] = []
    pr_bundles: list[tuple] = []
    prevalence = 0.0
    model_npz = (
        ("mfrag_net_fraglength", GROUP_LABELS["mfrag_net_fraglength"], FRAG_METH_NPZ, 31),
        ("mfrag_net_end4motif", GROUP_LABELS["mfrag_net_end4motif"], MOTIF_NPZ, 6),
        ("mfrag_net_fusion", GROUP_LABELS["mfrag_net_fusion"], FOURCH_NPZ, 31),
    )
    for mkey, label, npz_path, rep in model_npz:
        y, prob = _load_cnn_pred(npz_path)
        if prevalence == 0.0:
            prevalence = float(y.mean())
        seed_i = args.seed + rep
        roc_c, pr_c, aucs, praucs = _bootstrap_curves(y, prob, args.n_bootstrap, seed_i)
        roc_bundles.append((mkey, label, roc_c, aucs))
        pr_bundles.append((mkey, label, pr_c, praucs, prevalence))

    _plot_roc(
        roc_bundles,
        out_png=args.out_dir / "figure_s5_mfragnet_three_model_roc.png",
        out_pdf=args.out_dir / "figure_s5_mfragnet_three_model_roc.pdf",
        out_csv=args.out_dir / "figure_s5_mfragnet_three_model_roc_summary.csv",
        fpr_points=args.fpr_points,
        dpi=args.dpi,
    )
    _plot_pr(
        pr_bundles,
        prevalence,
        out_png=args.out_dir / "figure_s5_mfragnet_three_model_prroc.png",
        out_pdf=args.out_dir / "figure_s5_mfragnet_three_model_prroc.pdf",
        out_csv=args.out_dir / "figure_s5_mfragnet_three_model_prroc_summary.csv",
        recall_points=args.recall_points,
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
