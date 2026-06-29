#!/usr/bin/env python3
"""
Figure S5 — DELFI (S/L ratio) vs mFrag-DELFI (Frag length).

Two bootstrap boxplots (Testing AUC; Testing PRAUC).
X-axis: Lasso | Ridge | RF; two boxes per family:
  baseline → DELFI (S/L ratio)
  mean     → mFrag-DELFI (Frag length)

Data: Mathios/Mathios_slratio_method/pca66_bootstrap_compare/bootstrap_draws.csv
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

PCA66_DRAWS = SL_DIR / "pca66_bootstrap_compare" / "bootstrap_draws.csv"
SL_TEST_PRED = SL_DIR / "pca66_l1_baseline" / "test_predictions.csv"

FAMILIES = ("l1", "l2", "rf")
FAMILY_LABELS = {"l1": "Lasso", "l2": "Ridge", "rf": "RF"}
FAMILY_FROM_DRAWS = {"Lasso": "l1", "Ridge": "l2", "RF": "rf"}
VARIANTS = (
    ("baseline", "delfi", "DELFI (S/L ratio)"),
    ("mean", "mfrag_delfi", "mFrag-DELFI (Frag length)"),
)
COLORS = {
    "delfi": "#1f77b4",
    "mfrag_delfi": "#ff7f0e",
}

FONT_BASE = 14
FONT_TITLE = 17
FONT_LABEL = 15
FONT_TICK = 13
FONT_LEGEND = 10
FONT_BRACKET = 10
BOX_WIDTH = 0.34
GROUP_CENTERS = np.array([2.0, 7.0, 12.0], dtype=float)
OFFSETS = np.array([-0.42, 0.42])


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


def _load_series() -> dict[tuple[str, str], dict[str, np.ndarray]]:
    pca = pd.read_csv(PCA66_DRAWS)
    data: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for fam_name, fam_key in FAMILY_FROM_DRAWS.items():
        for var_key, model_key, _ in VARIANTS:
            sub = pca[(pca["family"] == fam_name) & (pca["variant"] == var_key)]
            data[(fam_key, model_key)] = {
                "auc": sub["test_auc"].to_numpy(dtype=np.float64),
                "prauc": sub["test_prauc"].to_numpy(dtype=np.float64),
            }
    return data


def _load_sample_ids() -> list[str]:
    df = pd.read_csv(SL_TEST_PRED)
    return df["sample_id"].astype(str).tolist()


def _mwu_baseline_vs_mean(data: dict[tuple[str, str], dict[str, np.ndarray]], metric: str) -> pd.DataFrame:
    rows = []
    for fam in FAMILIES:
        base = data[(fam, "delfi")][metric]
        mean = data[(fam, "mfrag_delfi")][metric]
        res = mannwhitneyu(mean, base, alternative="greater", method="auto")
        rows.append(
            {
                "family": FAMILY_LABELS[fam],
                "family_key": fam,
                "metric": metric,
                "u_stat": float(res.statistic),
                "p_value": float(res.pvalue),
                "median_delfi": float(np.median(base)),
                "median_mfrag_delfi": float(np.median(mean)),
                "alternative": "mFrag-DELFI > DELFI (S/L ratio)",
                "sig": _sig_star(float(res.pvalue)),
            }
        )
    return pd.DataFrame(rows)


def _plot_family_boxplot(
    *,
    data: dict[tuple[str, str], dict[str, np.ndarray]],
    metric: str,
    metric_label: str,
    out_png: Path,
    ylim: tuple[float, float],
    bracket_y: float,
    family_bracket_y_offset: dict[str, float],
    dpi: int,
    mwu_df: pd.DataFrame,
    plot_title: str,
) -> None:
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

    model_keys = [v[1] for v in VARIANTS]
    positions: list[float] = []
    box_data: list[np.ndarray] = []
    pos_map: dict[tuple[str, str], float] = {}

    for gi, fam in enumerate(FAMILIES):
        for oi, model_key in enumerate(model_keys):
            pos = float(GROUP_CENTERS[gi] + OFFSETS[oi])
            positions.append(pos)
            box_data.append(data[(fam, model_key)][metric])
            pos_map[(fam, model_key)] = pos

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
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
    for i, model_key in enumerate(model_keys * len(FAMILIES)):
        bp["boxes"][i].set_facecolor(COLORS[model_key])
        bp["boxes"][i].set_alpha(0.35)

    rng = np.random.default_rng(7)
    y_lo, y_hi = ylim
    for fam in FAMILIES:
        for model_key in model_keys:
            pos = pos_map[(fam, model_key)]
            pts = np.asarray(data[(fam, model_key)][metric], dtype=np.float64)
            pts = pts[(pts >= y_lo) & (pts <= y_hi) & np.isfinite(pts)]
            if pts.size == 0:
                continue
            k = min(pts.size, 60)
            if k < pts.size:
                pts = pts[rng.choice(pts.size, size=k, replace=False)]
            base = COLORS[model_key]
            r, g, b = to_rgb(base) if mcolors.is_color_like(base) else (0.4, 0.4, 0.4)
            light = (0.55 + 0.45 * r, 0.55 + 0.45 * g, 0.55 + 0.45 * b)
            jitter = rng.uniform(-0.08, 0.08, size=pts.size)
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
    h = 0.015 * y_span
    text_pad = 0.003 * y_span
    line_gap = 0.022 * y_span
    for _, row in mwu_df.iterrows():
        fam = str(row["family_key"])
        y_line = bracket_y + family_bracket_y_offset.get(fam, 0.0)
        x1 = pos_map[(fam, "delfi")]
        x2 = pos_map[(fam, "mfrag_delfi")]
        if x1 > x2:
            x1, x2 = x2, x1
        x_mid = (x1 + x2) / 2.0
        sig = str(row["sig"])
        p_fmt = _format_p(float(row["p_value"]))
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

    ax.set_xticks(GROUP_CENTERS)
    ax.set_xticklabels([FAMILY_LABELS[f] for f in FAMILIES], fontweight="bold")
    ax.set_ylabel(metric_label, fontweight="bold")
    ax.set_title(plot_title, fontsize=FONT_TITLE, fontweight="bold")
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="both", which="both", width=1.2, labelsize=FONT_TICK)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

    left = GROUP_CENTERS[0] + OFFSETS[0] - BOX_WIDTH / 2 - 0.5
    right = GROUP_CENTERS[-1] + OFFSETS[-1] + BOX_WIDTH / 2 + 0.5
    ax.set_xlim(left, right)

    ax.legend(
        handles=[
            Patch(facecolor=COLORS[v[1]], edgecolor="black", linewidth=1.2, alpha=0.35, label=v[2])
            for v in VARIANTS
        ],
        loc="upper right",
        framealpha=0.95,
        edgecolor="0.75",
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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = _load_series()
    mwu_auc = _mwu_baseline_vs_mean(data, "auc")
    mwu_pr = _mwu_baseline_vs_mean(data, "prauc")
    mwu = pd.concat([mwu_auc, mwu_pr], ignore_index=True)
    mwu.to_csv(args.out_dir / "figure_s5_delfi_vs_mfragdelfi_bootstrap_mwu.csv", index=False)
    print(mwu.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    summary_rows = []
    long_rows = []
    for fam in FAMILIES:
        for var_key, model_key, model_label in VARIANTS:
            for mkey in ("auc", "prauc"):
                arr = data[(fam, model_key)][mkey]
                summary_rows.append(
                    {
                        "family": FAMILY_LABELS[fam],
                        "family_key": fam,
                        "variant": var_key,
                        "model_key": model_key,
                        "model_label": model_label,
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
                            "family": FAMILY_LABELS[fam],
                            "family_key": fam,
                            "variant": var_key,
                            "model_key": model_key,
                            "model_label": model_label,
                            "metric": mkey,
                            "bootstrap_idx": i,
                            "value": float(v),
                        }
                    )
    pd.DataFrame(summary_rows).to_csv(
        args.out_dir / "figure_s5_delfi_vs_mfragdelfi_bootstrap_summary.csv",
        index=False,
    )
    pd.DataFrame(long_rows).to_csv(
        args.out_dir / "figure_s5_delfi_vs_mfragdelfi_bootstrap_draws.csv",
        index=False,
    )

    plot_title = bootstrap_plot_title("overall", "All stages", _load_sample_ids())

    _plot_family_boxplot(
        data=data,
        metric="auc",
        metric_label="Testing AUC",
        out_png=args.out_dir / "figure_s5_delfi_vs_mfragdelfi_bootstrap_boxplot_auc.png",
        ylim=(0.6, 1.2),
        bracket_y=1.0,
        family_bracket_y_offset={"l1": 0.05, "l2": 0.02, "rf": 0.0},
        dpi=args.dpi,
        mwu_df=mwu_auc,
        plot_title=plot_title,
    )
    _plot_family_boxplot(
        data=data,
        metric="prauc",
        metric_label="Testing PRAUC",
        out_png=args.out_dir / "figure_s5_delfi_vs_mfragdelfi_bootstrap_boxplot_prauc.png",
        ylim=(0.0, 1.0),
        bracket_y=0.8,
        family_bracket_y_offset={"l1": 0.02, "rf": -0.1},
        dpi=args.dpi,
        mwu_df=mwu_pr,
        plot_title=plot_title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
