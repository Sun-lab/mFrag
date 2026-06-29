#!/usr/bin/env python3
"""
Figure S6 — detection depth bootstrap boxplots with median trend lines.

Four separate figures (plot-only from cached bootstrap draws):
  1. mFrag-Net (Frag length) — Testing AUC vs depth
  2. mFrag-Net (Frag length) — Testing PRAUC vs depth
  3. mFrag-Net (Motif)       — Testing AUC vs depth
  4. mFrag-Net (Motif)       — Testing PRAUC vs depth

X-axis: 0.1%, 1%, 5%, 40%, 100%.
Each depth: 1000× bootstrap boxplot; medians connected by a line.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

CFCRAFT = Path("/fh/working/sun_w/saishi_cui/cfCraft")
FIG4_DIR = CFCRAFT / "mFrag_Figures" / "Figure4"
OUT_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure S6"
CACHE_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure6" / "detection_depth_bootstrap"
FD_DIR = CFCRAFT / "Mathios" / "FDConv"

if str(FIG4_DIR) not in sys.path:
    sys.path.insert(0, str(FIG4_DIR))
from figure4_plot_titles import bootstrap_plot_title  # noqa: E402

DEPTHS = ("0.1%", "1%", "5%", "40%", "100%")
MODELS = (
    {
        "key": "fraglength",
        "label": "mFrag-Net (Frag length)",
        "color": "#9467bd",
        "full_bootstrap": FD_DIR / "frag_meth3ch_new_rep031_bootstrap_compare" / "bootstrap_draws.csv",
    },
    {
        "key": "motif",
        "label": "mFrag-Net (Motif)",
        "color": "#ff7f0e",
        "full_bootstrap": FD_DIR / "motif_meth3ch_new_rep006_bootstrap_compare" / "bootstrap_draws.csv",
    },
)
PLOTS = (
    ("fraglength", "auc", "Testing AUC", "testing_auc"),
    ("fraglength", "prauc", "Testing PRAUC", "testing_prauc"),
    ("motif", "auc", "Testing AUC", "testing_auc"),
    ("motif", "prauc", "Testing PRAUC", "testing_prauc"),
)
PLOT_YLIM: dict[tuple[str, str], tuple[float, float]] = {
    ("fraglength", "auc"): (0.5, 1.0),
    ("fraglength", "prauc"): (0.0, 1.0),
    ("motif", "auc"): (0.4, 1.0),
    ("motif", "prauc"): (0.0, 1.0),
}

FONT_BASE = 14
FONT_TITLE = 17
FONT_LABEL = 15
FONT_TICK = 11
BOX_WIDTH = 0.42


def _load_depth_data(model_cfg: dict, cache_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for depth in DEPTHS:
        if depth == "100%":
            df = pd.read_csv(model_cfg["full_bootstrap"])
        else:
            df = pd.read_csv(cache_dir / model_cfg["key"] / depth / "bootstrap_draws.csv")
        out[depth] = {
            "auc": df["test_auc"].to_numpy(dtype=np.float64),
            "prauc": df["test_prauc"].to_numpy(dtype=np.float64),
        }
    return out


def _plot_depth_trend_boxplot(
    data: dict[str, np.ndarray],
    *,
    depths: tuple[str, ...],
    metric_label: str,
    model_label: str,
    color: str,
    plot_title: str,
    out_png: Path,
    out_pdf: Path,
    ylim: tuple[float, float],
    dpi: int,
) -> None:
    positions = np.arange(len(depths), dtype=float) + 1.0
    medians = np.array([float(np.median(data[d])) for d in depths], dtype=np.float64)

    mpl.rcParams.update(
        {
            "font.size": FONT_BASE,
            "axes.titlesize": FONT_TITLE,
            "axes.labelsize": FONT_LABEL,
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

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    bp = ax.boxplot(
        [data[d] for d in depths],
        positions=positions,
        widths=BOX_WIDTH,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 2.4},
        whiskerprops={"color": "black", "linewidth": 1.5},
        capprops={"color": "black", "linewidth": 1.5},
        boxprops={"edgecolor": "black", "linewidth": 1.5},
    )
    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(0.35)

    rng = np.random.default_rng(7)
    y_lo, y_hi = ylim
    r, g, b = to_rgb(color)
    light = (0.55 + 0.45 * r, 0.55 + 0.45 * g, 0.55 + 0.45 * b)
    for pos, depth in zip(positions, depths):
        pts = np.asarray(data[depth], dtype=np.float64)
        pts = pts[(pts >= y_lo) & (pts <= y_hi) & np.isfinite(pts)]
        if pts.size == 0:
            continue
        k = min(pts.size, 50)
        if k < pts.size:
            pts = pts[rng.choice(pts.size, size=k, replace=False)]
        jitter = rng.uniform(-0.06, 0.06, size=pts.size)
        ax.scatter(
            np.full_like(pts, pos) + jitter,
            pts,
            s=16,
            color=light,
            alpha=0.5,
            linewidths=0.0,
            zorder=3,
        )

    ax.plot(
        positions,
        medians,
        color=color,
        linewidth=2.8,
        marker="o",
        markersize=9,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=2.2,
        zorder=5,
        label="Median trend",
    )
    for x, y in zip(positions, medians):
        ax.scatter(x, y, s=70, color=color, edgecolors="black", linewidths=1.0, zorder=6)

    ax.set_xticks(positions)
    ax.set_xticklabels(depths, fontweight="bold")
    ax.set_xlabel("Detection depth (cfDNA fragment fraction)", fontweight="bold")
    ax.set_ylabel(metric_label, fontweight="bold")
    ax.set_title(f"{plot_title}\n{model_label}", fontweight="bold")
    ax.set_ylim(*ylim)
    ax.set_xlim(positions[0] - 0.65, positions[-1] + 0.65)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="both", which="both", width=1.2, labelsize=FONT_TICK)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR_DEFAULT)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    sample_ids = sorted(
        p.stem
        for p in (CFCRAFT / "Mathios" / "Final_data_24" / "testing" / "testing_fragment_length_zscored").glob(
            "*.csv"
        )
    )
    plot_title = bootstrap_plot_title("overall", "All stages", sample_ids)

    model_data = {m["key"]: _load_depth_data(m, args.cache_dir) for m in MODELS}
    model_by_key = {m["key"]: m for m in MODELS}

    summary_rows = []
    for model_key, metric_key, metric_label, fname_suffix in PLOTS:
        mcfg = model_by_key[model_key]
        depth_data = model_data[model_key]
        metric_arrays = {d: depth_data[d][metric_key] for d in DEPTHS}
        ylim = PLOT_YLIM[(model_key, metric_key)]

        for depth in DEPTHS:
            arr = metric_arrays[depth]
            summary_rows.append(
                {
                    "model_key": model_key,
                    "model_label": mcfg["label"],
                    "depth": depth,
                    "metric": metric_key,
                    "n": len(arr),
                    "median": float(np.median(arr)),
                    "q25": float(np.percentile(arr, 25)),
                    "q75": float(np.percentile(arr, 75)),
                }
            )

        _plot_depth_trend_boxplot(
            metric_arrays,
            depths=DEPTHS,
            metric_label=metric_label,
            model_label=mcfg["label"],
            color=mcfg["color"],
            plot_title=plot_title,
            out_png=args.out_dir / f"figure_s6_detection_depth_{model_key}_{fname_suffix}_boxplot.png",
            out_pdf=args.out_dir / f"figure_s6_detection_depth_{model_key}_{fname_suffix}_boxplot.pdf",
            ylim=ylim,
            dpi=args.dpi,
        )

    pd.DataFrame(summary_rows).to_csv(
        args.out_dir / "figure_s6_detection_depth_bootstrap_summary.csv",
        index=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
