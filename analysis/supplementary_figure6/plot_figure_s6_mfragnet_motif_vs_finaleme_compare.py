#!/usr/bin/env python3
"""
Figure S6 — mFrag-Net (Motif) with / without DNA methyl channels.

Bootstrap boxplot (Testing AUC / PRAUC) for three models:
  motif-only | referenced DNA methyl | FinaleMe inferred DNA methyl

Data:
  end4motif_new_rep093_bootstrap_compare
  motif_meth3ch_new_rep006_bootstrap_compare
  motif_finaleme_2ch_new_rep164_bootstrap_compare
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
FIG4_DIR = CFCRAFT / "mFrag_Figures" / "Figure4"
OUT_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure S6"

if str(FIG4_DIR) not in sys.path:
    sys.path.insert(0, str(FIG4_DIR))
from figure4_plot_titles import bootstrap_plot_title  # noqa: E402

MOTIF_ONLY_DRAWS = FD_DIR / "end4motif_new_rep093_bootstrap_compare" / "bootstrap_draws.csv"
MOTIF_METH_DRAWS = FD_DIR / "motif_meth3ch_new_rep006_bootstrap_compare" / "bootstrap_draws.csv"
MOTIF_FINALEME_DRAWS = FD_DIR / "motif_finaleme_2ch_new_rep164_bootstrap_compare" / "bootstrap_draws.csv"
MOTIF_METH_NPZ = FD_DIR / "motif_meth3ch_new" / "runs" / "rep_006.npz"

GROUPS = (
    ("mfrag_net_motifonly", "mFrag-Net (Motif) (without DNA Methyl)"),
    (
        "mfrag_net_referenced",
        "mFrag-Net (Motif) (with referenced DNA Methyl) (proposed model)",
    ),
    (
        "mfrag_net_inferred",
        "mFrag-Net (Motif) (with FinaleMe inferred DNA Methyl)",
    ),
)
GROUP_LABELS = {k: v for k, v in GROUPS}
COLORS = {
    "mfrag_net_motifonly": "#1f77b4",
    "mfrag_net_referenced": "#9467bd",
    "mfrag_net_inferred": "#ff7f0e",
}
PAIR_KEYS = (
    ("mfrag_net_motifonly", "mfrag_net_referenced"),
    ("mfrag_net_motifonly", "mfrag_net_inferred"),
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
    motifonly = pd.read_csv(MOTIF_ONLY_DRAWS)
    motif_meth = pd.read_csv(MOTIF_METH_DRAWS)
    finaleme = pd.read_csv(MOTIF_FINALEME_DRAWS)
    return {
        "mfrag_net_motifonly": {
            "auc": motifonly["test_auc"].to_numpy(dtype=np.float64),
            "prauc": motifonly["test_prauc"].to_numpy(dtype=np.float64),
        },
        "mfrag_net_referenced": {
            "auc": motif_meth["test_auc"].to_numpy(dtype=np.float64),
            "prauc": motif_meth["test_prauc"].to_numpy(dtype=np.float64),
        },
        "mfrag_net_inferred": {
            "auc": finaleme["test_auc"].to_numpy(dtype=np.float64),
            "prauc": finaleme["test_prauc"].to_numpy(dtype=np.float64),
        },
    }


def _load_sample_ids() -> list[str]:
    z = np.load(MOTIF_METH_NPZ, allow_pickle=True)
    return [str(f).replace(".csv", "") for f in z["sample_files"]]


def _write_metric_summaries(data: dict[str, dict[str, np.ndarray]], out_dir: Path) -> None:
    roc_rows = []
    pr_rows = []
    for gkey, glabel in GROUPS:
        auc = data[gkey]["auc"]
        prauc = data[gkey]["prauc"]
        roc_rows.append(
            {
                "model_key": gkey,
                "label": glabel,
                "n_bootstrap": len(auc),
                "auc_median": float(np.median(auc)),
                "auc_q25": float(np.percentile(auc, 25)),
                "auc_q75": float(np.percentile(auc, 75)),
            }
        )
        pr_rows.append(
            {
                "model_key": gkey,
                "label": glabel,
                "n_bootstrap": len(prauc),
                "prauc_median": float(np.median(prauc)),
                "prauc_q25": float(np.percentile(prauc, 25)),
                "prauc_q75": float(np.percentile(prauc, 75)),
            }
        )
    pd.DataFrame(roc_rows).to_csv(out_dir / "figure_s6_mfragnet_motif_vs_finaleme_roc_summary.csv", index=False)
    pd.DataFrame(pr_rows).to_csv(out_dir / "figure_s6_mfragnet_motif_vs_finaleme_prroc_summary.csv", index=False)


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
    plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    for path in (MOTIF_ONLY_DRAWS, MOTIF_METH_DRAWS, MOTIF_FINALEME_DRAWS):
        if not path.is_file():
            raise SystemExit(f"Missing {path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = _load_series()
    mwu_auc = _mwu_pairwise(data, "auc")
    mwu_pr = _mwu_pairwise(data, "prauc")
    mwu = pd.concat([mwu_auc, mwu_pr], ignore_index=True)
    mwu.to_csv(args.out_dir / "figure_s6_mfragnet_motif_vs_finaleme_bootstrap_mwu.csv", index=False)
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
        args.out_dir / "figure_s6_mfragnet_motif_vs_finaleme_bootstrap_summary.csv",
        index=False,
    )
    pd.DataFrame(long_rows).to_csv(
        args.out_dir / "figure_s6_mfragnet_motif_vs_finaleme_bootstrap_draws.csv",
        index=False,
    )
    _write_metric_summaries(data, args.out_dir)

    plot_title = bootstrap_plot_title("overall", "All stages", _load_sample_ids())
    _plot_boxplot(
        data=data,
        out_png=args.out_dir / "figure_s6_mfragnet_motif_vs_finaleme_bootstrap_boxplot.png",
        ylim=(0.0, 1.4),
        dpi=args.dpi,
        mwu_auc=mwu_auc,
        mwu_pr=mwu_pr,
        plot_title=plot_title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
