#!/usr/bin/env python3
"""
Figure S6 — Testing AUC / PRAUC bootstrap boxplot (four models).

1. mFrag-Weight (S/L ratio)   — PCA66 Lasso |Δβ|/mean (pca66_bootstrap_compare)
2. mFrag-Net (Frag length)    — frag_meth3ch_new rep031 bootstrap
3. Fragle (ctDNA %)           — 1000× bootstrap on testing Fragle ctDNA fractions
4. ichorCNA (ctDNA %)         — 1000× bootstrap on testing ichorCNA tumor fractions
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
from sklearn.metrics import average_precision_score, roc_auc_score

CFCRAFT = Path("/fh/working/sun_w/saishi_cui/cfCraft")
MATHIOS = CFCRAFT / "Mathios"
SL_DIR = MATHIOS / "Mathios_slratio_method"
FD_DIR = MATHIOS / "FDConv"
FRAGLE_DIR = CFCRAFT / "Fragle"
FIG4_DIR = CFCRAFT / "mFrag_Figures" / "Figure4"
OUT_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure S6"

if str(FIG4_DIR) not in sys.path:
    sys.path.insert(0, str(FIG4_DIR))
from figure4_plot_titles import bootstrap_plot_title  # noqa: E402

PCA66_DRAWS = SL_DIR / "pca66_bootstrap_compare" / "bootstrap_draws.csv"
MFRAGNET_DRAWS = FD_DIR / "frag_meth3ch_new_rep031_bootstrap_compare" / "bootstrap_draws.csv"
FRAGLE_SUMMARY = FRAGLE_DIR / "fragle_ctdna_predictions_summary.csv"
FRAGLE_MANIFEST = FRAGLE_DIR / "bam_manifest.tsv"
ICHOR_CSV = MATHIOS / "ichorCNA_results" / "ctDNA_fraction_all_samples.csv"
GROUPS = (
    ("mfrag_delfi", "mFrag-Weight (S/L ratio)"),
    ("mfrag_net", "mFrag-Net (Frag length)"),
    ("fragle", "Fragle (ctDNA %)"),
    ("ichorcna", "ichorCNA (ctDNA %)"),
)
COLORS = {
    "mfrag_delfi": "#ff7f0e",
    "mfrag_net": "#9467bd",
    "fragle": "#2ca02c",
    "ichorcna": "#8c564b",
}
GROUP_LABELS = {k: v for k, v in GROUPS}
# Compare model 1 vs 3 and model 2 vs 3 (each vs Fragle).
PAIR_KEYS = (
    ("mfrag_delfi", "fragle"),
    ("mfrag_net", "fragle"),
)

FONT_BASE = 14
FONT_TITLE = 17
FONT_LABEL = 15
FONT_TICK = 13
FONT_LEGEND = 10
FONT_BRACKET = 11
BOX_SPAN = 0.38
BOX_WIDTH = 0.08
BOX_XPAD = 0.20
METRIC_CENTERS = (1.0, 2.0)
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


def _bootstrap_testing(
    y: np.ndarray,
    score: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs: list[float] = []
    praucs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        s_b = score[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aucs.append(float(roc_auc_score(y_b, s_b)))
        praucs.append(float(average_precision_score(y_b, s_b)))
    return np.asarray(aucs, dtype=np.float64), np.asarray(praucs, dtype=np.float64)


def _load_mfrag_delfi() -> dict[str, np.ndarray]:
    df = pd.read_csv(PCA66_DRAWS)
    sub = df[(df["family"] == "Lasso") & (df["variant"] == "mean")]
    return {
        "auc": sub["test_auc"].to_numpy(dtype=np.float64),
        "prauc": sub["test_prauc"].to_numpy(dtype=np.float64),
    }


def _load_mfrag_net() -> dict[str, np.ndarray]:
    df = pd.read_csv(MFRAGNET_DRAWS)
    return {
        "auc": df["test_auc"].to_numpy(dtype=np.float64),
        "prauc": df["test_prauc"].to_numpy(dtype=np.float64),
    }


def _load_fragle_testing(n_boot: int, seed: int) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    summary = pd.read_csv(FRAGLE_SUMMARY)
    manifest = pd.read_csv(FRAGLE_MANIFEST, sep="\t")
    df = summary.merge(manifest[["sample_id", "split"]], on="sample_id", how="inner")
    df = df[(df["split"] == "testing") & (df["group"].isin(["healthy", "cancer"]))].copy()
    y = (df["group"] == "cancer").astype(int).to_numpy()
    score = df["ctdna_fraction"].to_numpy(dtype=float)
    aucs, praucs = _bootstrap_testing(y, score, n_boot, seed)
    draws = pd.DataFrame(
        {
            "bootstrap_idx": np.arange(len(aucs)),
            "test_auc": aucs,
            "test_prauc": praucs,
        }
    )
    return {"auc": aucs, "prauc": praucs}, draws


def _load_ichorcna_testing(n_boot: int, seed: int) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    df = pd.read_csv(ICHOR_CSV)
    df = df[
        (df["split"] == "testing")
        & (df["status"] == "ok")
        & df["ctDNA_fraction"].notna()
        & df["label"].isin(["healthy", "lung_cancer"])
    ].copy()
    y = (df["label"] == "lung_cancer").astype(int).to_numpy()
    score = df["ctDNA_fraction"].to_numpy(dtype=float)
    aucs, praucs = _bootstrap_testing(y, score, n_boot, seed)
    draws = pd.DataFrame(
        {
            "bootstrap_idx": np.arange(len(aucs)),
            "test_auc": aucs,
            "test_prauc": praucs,
        }
    )
    return {"auc": aucs, "prauc": praucs}, draws


def _load_testing_sample_ids() -> list[str]:
    meta = pd.read_csv(FRAGLE_MANIFEST, sep="\t")
    return meta.loc[meta["split"] == "testing", "sample_id"].astype(str).tolist()


def _plot_boxplot(
    data: dict[str, dict[str, np.ndarray]],
    *,
    out_png: Path,
    out_pdf: Path,
    ylim: tuple[float, float],
    dpi: int,
    plot_title: str,
    mwu_auc: pd.DataFrame,
    mwu_pr: pd.DataFrame,
) -> None:
    metric_keys = ("auc", "prauc")
    metric_labels = ("Testing AUC", "Testing PRAUC")
    group_centers = np.array(METRIC_CENTERS, dtype=float)
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
        gc = group_centers[mi]
        for gi, (gkey, _) in enumerate(GROUPS):
            pos_map[(mkey, gkey)] = gc + offsets[gi]
            box_data.append(data[gkey][mkey])
            colors_order.append(gkey)

    fig, ax = plt.subplots(figsize=(12.5, 6.8))
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
            jitter = rng.uniform(-0.03, 0.03, size=pts.size)
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
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
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
    fragle_draws, fragle_boot = _load_fragle_testing(args.n_bootstrap, args.seed)
    ichor_draws, ichor_boot = _load_ichorcna_testing(args.n_bootstrap, args.seed + 99)
    fragle_boot.to_csv(args.out_dir / "figure_s6_fragle_testing_bootstrap_draws.csv", index=False)
    ichor_boot.to_csv(args.out_dir / "figure_s6_ichorcna_testing_bootstrap_draws.csv", index=False)

    data = {
        "mfrag_delfi": _load_mfrag_delfi(),
        "mfrag_net": _load_mfrag_net(),
        "fragle": fragle_draws,
        "ichorcna": ichor_draws,
    }

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
        args.out_dir / "figure_s6_mfragdelfi_mfragnet_fragle_bootstrap_summary.csv",
        index=False,
    )
    pd.DataFrame(long_rows).to_csv(
        args.out_dir / "figure_s6_mfragdelfi_mfragnet_fragle_bootstrap_draws.csv",
        index=False,
    )

    plot_title = bootstrap_plot_title("overall", "All stages", _load_testing_sample_ids())

    mwu_auc = _mwu_pairwise(data, "auc")
    mwu_pr = _mwu_pairwise(data, "prauc")
    mwu = pd.concat([mwu_auc, mwu_pr], ignore_index=True)
    mwu.to_csv(args.out_dir / "figure_s6_mfragdelfi_mfragnet_fragle_bootstrap_mwu.csv", index=False)
    print(mwu.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    _plot_boxplot(
        data,
        out_png=args.out_dir / "figure_s6_mfragdelfi_mfragnet_fragle_testing_bootstrap_boxplot.png",
        out_pdf=args.out_dir / "figure_s6_mfragdelfi_mfragnet_fragle_testing_bootstrap_boxplot.pdf",
        ylim=(0.0, 1.5),
        dpi=args.dpi,
        plot_title=plot_title,
        mwu_auc=mwu_auc,
        mwu_pr=mwu_pr,
    )

    print(pd.DataFrame(summary_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
