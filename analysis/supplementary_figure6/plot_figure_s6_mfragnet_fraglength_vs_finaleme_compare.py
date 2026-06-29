#!/usr/bin/env python3
"""
Figure S6 — mFrag-Net (Frag length) with / without DNA methyl channels.

Bootstrap boxplot (Testing AUC / PRAUC) and paired ROC / PR-ROC curves.

Data:
  fragonly_new_rep135_bootstrap_compare
  frag_meth3ch_new_rep031_bootstrap_compare
  frag_finaleme_2ch_new_rep200_bootstrap_compare
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
OUT_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure S6"

if str(MATHIOS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MATHIOS_SCRIPTS))
if str(FIG4_DIR) not in sys.path:
    sys.path.insert(0, str(FIG4_DIR))
from figure4_plot_titles import bootstrap_plot_title  # noqa: E402
from roc_plot_utils import pad_median_iqr_for_origin_vertical, roc_median_iqr  # noqa: E402

FRAGONLY_DRAWS = FD_DIR / "fragonly_new_rep135_bootstrap_compare" / "bootstrap_draws.csv"
FRAG_METH_DRAWS = FD_DIR / "frag_meth3ch_new_rep031_bootstrap_compare" / "bootstrap_draws.csv"
FINALEME_DRAWS = FD_DIR / "frag_finaleme_2ch_new_rep200_bootstrap_compare" / "bootstrap_draws.csv"
FRAGONLY_NPZ = FD_DIR / "fragonly_new" / "runs" / "rep_135.npz"
FRAG_METH_NPZ = FD_DIR / "frag_meth3ch_new" / "runs" / "rep_031.npz"
FINALEME_NPZ = FD_DIR / "frag_finaleme_2ch_new" / "runs" / "rep_200.npz"

GROUPS = (
    ("mfrag_net_fragonly", "mFrag-Net (Frag length) (without DNA Methyl)"),
    (
        "mfrag_net_referenced",
        "mFrag-Net (Frag length) (with referenced DNA Methyl) (proposed model)",
    ),
    (
        "mfrag_net_inferred",
        "mFrag-Net (Frag length) (with FinaleMe inferred DNA Methyl)",
    ),
)
GROUP_LABELS = {k: v for k, v in GROUPS}
COLORS = {
    "mfrag_net_fragonly": "#1f77b4",
    "mfrag_net_referenced": "#9467bd",
    "mfrag_net_inferred": "#ff7f0e",
}
# Compare model 1 vs 2 and model 1 vs 3.
PAIR_KEYS = (
    ("mfrag_net_fragonly", "mfrag_net_referenced"),
    ("mfrag_net_fragonly", "mfrag_net_inferred"),
)

FONT_BASE = 14
FONT_TITLE = 17
FONT_LABEL = 15
FONT_TICK = 13
FONT_LEGEND = 10
FONT_BRACKET = 11
BOX_SPAN = 0.34
BOX_WIDTH = 0.10
BOX_XPAD = 0.22
BRACKET_BLOCK_FRAC = 0.082
BRACKET_TIER_GAP_FRAC = 0.016
BRACKET_TOP_MARGIN_FRAC = 0.024


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


def _sorted_mwu_for_brackets(
    mwu_df: pd.DataFrame,
    pos_map: dict[tuple[str, str], float],
    metric: str,
) -> pd.DataFrame:
    rows = []
    for _, row in mwu_df.iterrows():
        x1 = pos_map[(metric, row["group_a_key"])]
        x2 = pos_map[(metric, row["group_b_key"])]
        rows.append({**row.to_dict(), "_x_span": abs(x2 - x1)})
    return pd.DataFrame(rows).sort_values("_x_span", ascending=False).reset_index(drop=True)


def _bracket_y_levels(*, n_brackets: int, data_top: float, y_span: float) -> list[float]:
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


def _load_series() -> dict[str, dict[str, np.ndarray]]:
    fragonly = pd.read_csv(FRAGONLY_DRAWS)
    fmeth = pd.read_csv(FRAG_METH_DRAWS)
    finaleme = pd.read_csv(FINALEME_DRAWS)
    return {
        "mfrag_net_fragonly": {
            "auc": fragonly["test_auc"].to_numpy(dtype=np.float64),
            "prauc": fragonly["test_prauc"].to_numpy(dtype=np.float64),
        },
        "mfrag_net_referenced": {
            "auc": fmeth["test_auc"].to_numpy(dtype=np.float64),
            "prauc": fmeth["test_prauc"].to_numpy(dtype=np.float64),
        },
        "mfrag_net_inferred": {
            "auc": finaleme["test_auc"].to_numpy(dtype=np.float64),
            "prauc": finaleme["test_prauc"].to_numpy(dtype=np.float64),
        },
    }


def _load_sample_ids() -> list[str]:
    z = np.load(FRAG_METH_NPZ, allow_pickle=True)
    return [str(f).replace(".csv", "") for f in z["sample_files"]]


def _plot_boxplot(
    *,
    data: dict[str, dict[str, np.ndarray]],
    out_png: Path,
    out_pdf: Path,
    ylim: tuple[float, float],
    dpi: int,
    mwu_auc: pd.DataFrame,
    mwu_pr: pd.DataFrame,
    plot_title: str,
) -> None:
    metric_keys = ("auc", "prauc")
    metric_labels = ("Testing AUC", "Testing PRAUC")
    group_centers = np.array([1.0, 2.0], dtype=float)
    offsets = np.linspace(-BOX_SPAN / 2, BOX_SPAN / 2, len(GROUPS))
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
            box_data.append(data[gkey][mkey])
            colors_order.append(gkey)
            pos_map[(mkey, gkey)] = pos

    fig, ax = plt.subplots(figsize=(11.0, 6.8))
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
        panel_vals = np.concatenate([data[gkey][mkey] for gkey, _ in GROUPS])
        data_top = float(np.max(panel_vals[np.isfinite(panel_vals)]))
        ordered = _sorted_mwu_for_brackets(mwu_df, pos_map, mkey)
        levels = _bracket_y_levels(n_brackets=len(ordered), data_top=data_top, y_span=y_span)
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


def _pr_median_iqr(curves, recall_grid):
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
    bundles,
    *,
    title: str,
    out_png: Path,
    out_pdf: Path,
    out_csv: Path,
    fpr_points: int,
    dpi: int,
) -> None:
    fpr_grid = np.linspace(0.0, 1.0, fpr_points)
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.35, zorder=0)
    rows = []
    for mkey, label, roc_curves, aucs in bundles:
        med, lo, hi = roc_median_iqr(roc_curves, fpr_grid)
        fpr_plot, med, lo, hi = pad_median_iqr_for_origin_vertical(fpr_grid, med, lo, hi)
        auc_med = float(np.median(aucs))
        auc_q1 = float(np.percentile(aucs, 25))
        auc_q3 = float(np.percentile(aucs, 75))
        ax.fill_between(fpr_plot, lo, hi, color=COLORS[mkey], alpha=0.18, linewidth=0)
        ax.plot(
            fpr_plot,
            med,
            color=COLORS[mkey],
            lw=2.4,
            label=f"{label} ({auc_med:.3f} [{auc_q1:.3f}, {auc_q3:.3f}])",
            zorder=2,
        )
        rows.append(
            {
                "model_key": mkey,
                "label": label,
                "n_bootstrap": len(aucs),
                "auc_median": auc_med,
                "auc_q25": auc_q1,
                "auc_q75": auc_q3,
            }
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("True positive rate", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
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
    print(f"Wrote {out_pdf}")


def _plot_pr(
    bundles,
    prevalence: float,
    *,
    title: str,
    out_png: Path,
    out_pdf: Path,
    out_csv: Path,
    recall_points: int,
    dpi: int,
) -> None:
    recall_grid = np.linspace(0.0, 1.0, recall_points)
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    rows = []
    for mkey, label, pr_curves, praucs, _ in bundles:
        med, lo, hi = _pr_median_iqr(pr_curves, recall_grid)
        pr_med = float(np.median(praucs))
        pr_q1 = float(np.percentile(praucs, 25))
        pr_q3 = float(np.percentile(praucs, 75))
        ax.fill_between(recall_grid, lo, hi, color=COLORS[mkey], alpha=0.18, linewidth=0)
        ax.plot(
            recall_grid,
            med,
            color=COLORS[mkey],
            lw=2.4,
            label=f"{label} ({pr_med:.3f} [{pr_q1:.3f}, {pr_q3:.3f}])",
            zorder=2,
        )
        rows.append(
            {
                "model_key": mkey,
                "label": label,
                "n_bootstrap": len(praucs),
                "prauc_median": pr_med,
                "prauc_q25": pr_q1,
                "prauc_q75": pr_q3,
            }
        )
    ax.axhline(prevalence, color="k", ls="--", lw=1, alpha=0.4, label=f"prevalence = {prevalence:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
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
    print(f"Wrote {out_pdf}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = _load_series()
    mwu_auc = _mwu_pairwise(data, "auc")
    mwu_pr = _mwu_pairwise(data, "prauc")
    mwu = pd.concat([mwu_auc, mwu_pr], ignore_index=True)
    mwu.to_csv(args.out_dir / "figure_s6_mfragnet_fraglength_vs_finaleme_bootstrap_mwu.csv", index=False)
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
        args.out_dir / "figure_s6_mfragnet_fraglength_vs_finaleme_bootstrap_summary.csv",
        index=False,
    )
    pd.DataFrame(long_rows).to_csv(
        args.out_dir / "figure_s6_mfragnet_fraglength_vs_finaleme_bootstrap_draws.csv",
        index=False,
    )

    sample_ids = _load_sample_ids()
    plot_title = bootstrap_plot_title("overall", "All stages", sample_ids)

    _plot_boxplot(
        data=data,
        out_png=args.out_dir / "figure_s6_mfragnet_fraglength_vs_finaleme_bootstrap_boxplot.png",
        out_pdf=args.out_dir / "figure_s6_mfragnet_fraglength_vs_finaleme_bootstrap_boxplot.pdf",
        ylim=(0.0, 1.4),
        dpi=args.dpi,
        mwu_auc=mwu_auc,
        mwu_pr=mwu_pr,
        plot_title=plot_title,
    )

    roc_bundles, pr_bundles = [], []
    prevalence = 0.0
    model_npz = (
        ("mfrag_net_fragonly", GROUP_LABELS["mfrag_net_fragonly"], FRAGONLY_NPZ, 135),
        ("mfrag_net_referenced", GROUP_LABELS["mfrag_net_referenced"], FRAG_METH_NPZ, 31),
        ("mfrag_net_inferred", GROUP_LABELS["mfrag_net_inferred"], FINALEME_NPZ, 200),
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
        title=plot_title,
        out_png=args.out_dir / "figure_s6_mfragnet_fraglength_vs_finaleme_roc.png",
        out_pdf=args.out_dir / "figure_s6_mfragnet_fraglength_vs_finaleme_roc.pdf",
        out_csv=args.out_dir / "figure_s6_mfragnet_fraglength_vs_finaleme_roc_summary.csv",
        fpr_points=2001,
        dpi=args.dpi,
    )
    _plot_pr(
        pr_bundles,
        prevalence,
        title=plot_title,
        out_png=args.out_dir / "figure_s6_mfragnet_fraglength_vs_finaleme_prroc.png",
        out_pdf=args.out_dir / "figure_s6_mfragnet_fraglength_vs_finaleme_prroc.pdf",
        out_csv=args.out_dir / "figure_s6_mfragnet_fraglength_vs_finaleme_prroc_summary.csv",
        recall_points=2001,
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
