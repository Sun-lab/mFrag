#!/usr/bin/env python3
"""Figure 2: 2×2 panels — S/L boxplots (row 1) and external-ref β barplots (row 2)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

MATHIOS = Path(__file__).resolve().parents[1]
FRAG_DIR = Path(__file__).resolve().parent
CFCRAFT = MATHIOS.parent
FIGURE2_DIR = CFCRAFT / "mFrag_Figures" / "Figure2"

SL_GENOME_CSV = FRAG_DIR / "sl_ratio_genomewide_100_220_samples.csv"
SL_ZSCORE_CSV = MATHIOS / "Mathios_slratio_method" / "slratio_training_row_zscore.csv"
EXT_METH_CSV = MATHIOS / "External_DNA_Methylation" / "external_window_mean_beta_ttest.csv"
H_BETA_DIR = MATHIOS / "Final_data_24" / "training" / "training_DNAMethylation_HealthyRef_raw_weights"
L_BETA_DIR = MATHIOS / "Final_data_24" / "training" / "training_DNAMethylation_LungcancerRef_raw_weights"
BETA_CACHE = FRAG_DIR / "figure2_panel_beta_by_region.npz"

DEFAULT_OUT_PNG = FIGURE2_DIR / "figure2_sl_beta_boxplot_panels.png"
DEFAULT_OUT_PDF = FIGURE2_DIR / "figure2_sl_beta_boxplot_panels.pdf"

REGIONS = (
    {"key": "genomewide", "title": "Genome-wide", "window_id": None},
    {"key": "chr5_5_10", "title": "Chr5:5–10 Mb", "window_id": "chr5_5000000_10000000"},
)

BETA_YLIM = (0.4, 1.0)
COLOR_HEALTHY = "#2166AC"
COLOR_CANCER = "#B2182B"
FILL_HEALTHY = "#92C5DE"
FILL_CANCER = "#F4A582"
BOX_LW = 2.0
BAR_LW = 1.8
FONT_TITLE = 14
FONT_LABEL = 13
FONT_TICK = 12
FONT_ANNOT = 11
FONT_DELTA = 12
CANCER_RATIO_CAP = 1.0
BETA_CENTERS = np.array([(i + 0.5) / 24.0 for i in range(24)], dtype=np.float64)


BAR_ERR_FLOOR = 0.028  # minimum visible error-bar half-length on 0.4–1 axis
DELTA_BRACKET_X = 1.38


def _format_p(p: float, *, scientific: bool = False) -> str:
    if scientific:
        return f"{p:.2e}"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}".rstrip("0").rstrip(".")


def _format_p_exact(p: float) -> str:
    """Always show numeric p (scientific notation when very small)."""
    if not np.isfinite(p):
        return "NA"
    if p == 0.0:
        return "0"
    if p < 1e-4:
        return f"{p:.2e}"
    if p < 0.001:
        return f"{p:.4f}"
    return f"{p:.3f}".rstrip("0").rstrip(".")


def _weighted_mean_beta(row: np.ndarray) -> float:
    total = float(row.sum())
    if total <= 0 or not np.isfinite(total):
        return np.nan
    return float(np.dot(BETA_CENTERS, row) / total)


def _load_sl_genomewide() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(SL_GENOME_CSV)
    h = df.loc[df["group"] == "healthy", "short_long_ratio"].to_numpy(dtype=np.float64)
    c = df.loc[df["group"] == "lung_cancer", "short_long_ratio"].to_numpy(dtype=np.float64)
    h = h[np.isfinite(h)]
    c = c[np.isfinite(c)]
    c = c[c <= CANCER_RATIO_CAP]
    return h, c


def _load_sl_window_zscore(window_id: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(SL_ZSCORE_CSV)
    if window_id not in df.columns:
        raise SystemExit(f"Missing window column {window_id} in {SL_ZSCORE_CSV}")
    is_h = df["sample_id"].astype(str).str.endswith("_healthy")
    is_c = df["sample_id"].astype(str).str.endswith("_lung_cancer")
    h = df.loc[is_h, window_id].to_numpy(dtype=np.float64)
    c = df.loc[is_c, window_id].to_numpy(dtype=np.float64)
    return h[np.isfinite(h)], c[np.isfinite(c)]


def _beta_from_csv(path: Path, window_id: str | None) -> float:
    df = pd.read_csv(path)
    beta_cols = [c for c in df.columns if c != "window_id"]
    if window_id is None:
        pooled = df[beta_cols].sum(axis=0).to_numpy(dtype=np.float64)
    else:
        sub = df.loc[df["window_id"].astype(str) == window_id, beta_cols]
        if sub.empty:
            return np.nan
        pooled = sub.iloc[0].to_numpy(dtype=np.float64)
    return _weighted_mean_beta(pooled)


def _load_beta_by_region(
    rebuild: bool,
    *,
    regions: tuple[dict, ...] = REGIONS,
    cache_path: Path = BETA_CACHE,
) -> dict[str | None, tuple[np.ndarray, np.ndarray]]:
    keys = [r["window_id"] for r in regions]
    if cache_path.exists() and not rebuild:
        data = np.load(cache_path, allow_pickle=True)
        out: dict[str | None, tuple[np.ndarray, np.ndarray]] = {}
        for key in keys:
            tag = "genomewide" if key is None else key
            if f"{tag}_healthy" not in data:
                rebuild = True
                break
            out[key] = (data[f"{tag}_healthy"], data[f"{tag}_cancer"])
        if not rebuild:
            print(f"Loaded beta cache {cache_path}")
            return out

    out = {}
    for key in keys:
        h_vals, c_vals = [], []
        for path in sorted(H_BETA_DIR.glob("*_healthy.csv")):
            h_vals.append(_beta_from_csv(path, key))
        for path in sorted(L_BETA_DIR.glob("*_lung_cancer.csv")):
            c_vals.append(_beta_from_csv(path, key))
        h_arr = np.asarray(h_vals, dtype=np.float64)
        c_arr = np.asarray(c_vals, dtype=np.float64)
        out[key] = (h_arr[np.isfinite(h_arr)], c_arr[np.isfinite(c_arr)])

    save_kw = {}
    for key, (h_arr, c_arr) in out.items():
        tag = "genomewide" if key is None else key
        save_kw[f"{tag}_healthy"] = h_arr
        save_kw[f"{tag}_cancer"] = c_arr
    np.savez(cache_path, **save_kw)
    print(f"Wrote beta cache {cache_path}")
    return out


def _external_reference_means(window_id: str | None) -> tuple[float, float]:
    meth = pd.read_csv(EXT_METH_CSV)
    if window_id is not None:
        row = meth.loc[meth["window_id"].astype(str) == window_id]
        if row.empty:
            raise SystemExit(f"Missing external beta for {window_id}")
        row = row.iloc[0]
        return float(row["healthy_mean_beta"]), float(row["lung_cancer_mean_beta"])

    h_vals = meth["healthy_mean_beta"].to_numpy(dtype=np.float64)
    h_wts = meth["healthy_n_cpgs"].to_numpy(dtype=np.float64)
    l_vals = meth["lung_cancer_mean_beta"].to_numpy(dtype=np.float64)
    l_wts = meth["lung_cancer_n_cpgs"].to_numpy(dtype=np.float64)
    ok_h = np.isfinite(h_vals) & np.isfinite(h_wts) & (h_wts > 0)
    ok_l = np.isfinite(l_vals) & np.isfinite(l_wts) & (l_wts > 0)
    return float(np.average(h_vals[ok_h], weights=h_wts[ok_h])), float(
        np.average(l_vals[ok_l], weights=l_wts[ok_l])
    )


def _draw_boxplot_panel(
    ax,
    healthy: np.ndarray,
    cancer: np.ndarray,
    *,
    seed: int,
    title: str,
    ylim: tuple[float, float] | None,
    zscore_line: bool,
    p_scientific: bool,
) -> None:
    positions = [1, 2]
    data = [healthy, cancer]
    labels = ["Healthy", "Lung cancer"]

    box_kw = dict(
        patch_artist=True,
        widths=0.52,
        showfliers=False,
        boxprops=dict(linewidth=BOX_LW, edgecolor="black"),
        whiskerprops=dict(linewidth=BOX_LW, color="black"),
        capprops=dict(linewidth=BOX_LW, color="black"),
        medianprops=dict(color="black", linewidth=BOX_LW),
    )
    try:
        bp = ax.boxplot(data, positions=positions, tick_labels=labels, **box_kw)
    except TypeError:
        bp = ax.boxplot(data, positions=positions, labels=labels, **box_kw)

    for patch, fill in zip(bp["boxes"], (FILL_HEALTHY, FILL_CANCER)):
        patch.set_facecolor(fill)
        patch.set_edgecolor("black")
        patch.set_linewidth(BOX_LW)
        patch.set_alpha(0.88)

    rng = np.random.default_rng(seed)
    for pos, vals, color in zip(positions, data, (COLOR_HEALTHY, COLOR_CANCER)):
        jitter = pos + 0.10 * rng.standard_normal(len(vals))
        ax.scatter(jitter, vals, s=16, alpha=0.42, c=color, edgecolors="none", zorder=3)

    if len(healthy) >= 1 and len(cancer) >= 1:
        _, p = stats.mannwhitneyu(healthy, cancer, alternative="two-sided")
        ax.text(
            0.5,
            0.97,
            f"Mann–Whitney U, p = {_format_p(p, scientific=p_scientific)}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=FONT_ANNOT,
            fontweight="bold",
        )

    if zscore_line:
        ax.axhline(0, color="0.45", linestyle="--", linewidth=0.8, zorder=1)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=8)
    ax.grid(True, alpha=0.22, axis="y")
    ax.tick_params(axis="both", labelsize=FONT_TICK)


def _bar_error(vals: np.ndarray) -> float:
    """Per-bar error-bar half-length (sample SD, with a visibility floor)."""
    if len(vals) < 2:
        return BAR_ERR_FLOOR
    err = float(np.std(vals, ddof=1))
    return max(err, BAR_ERR_FLOOR)


def _annotate_delta_beta(
    ax,
    y_healthy: float,
    y_cancer: float,
    *,
    x_h: float = 0.0,
    x_c: float = 1.0,
) -> None:
    """Right-side vertical bracket showing Δβ between external reference bars."""
    y_top = max(y_healthy, y_cancer)
    y_bot = min(y_healthy, y_cancer)
    delta = y_top - y_bot
    x_br = DELTA_BRACKET_X
    cap = 0.045
    lw = 2.0
    guide = dict(color="0.35", linestyle="--", linewidth=1.0, zorder=3)

    ax.plot([x_h + 0.28, x_br], [y_healthy, y_healthy], **guide)
    ax.plot([x_c - 0.28, x_br], [y_cancer, y_cancer], **guide)
    ax.plot([x_br, x_br], [y_bot, y_top], color="black", linewidth=lw, zorder=4, solid_capstyle="round")
    ax.plot([x_br - cap, x_br + cap], [y_bot, y_bot], color="black", linewidth=lw, zorder=4)
    ax.plot([x_br - cap, x_br + cap], [y_top, y_top], color="black", linewidth=lw, zorder=4)
    ax.text(
        x_br + 0.07,
        0.5 * (y_bot + y_top),
        rf"$\Delta\beta = {delta:.2f}$",
        va="center",
        ha="left",
        fontsize=FONT_DELTA,
        fontweight="bold",
        color="black",
        zorder=5,
    )


def _draw_external_beta_bar_panel(
    ax,
    healthy: np.ndarray,
    cancer: np.ndarray,
    window_id: str | None,
    *,
    title: str,
) -> None:
    bar_h, bar_c = _external_reference_means(window_id)
    err_h = _bar_error(healthy)
    err_c = _bar_error(cancer)

    x = np.array([0, 1], dtype=np.float64)
    heights = np.array([bar_h, bar_c], dtype=np.float64)
    yerr = np.array([err_h, err_c], dtype=np.float64)
    colors = [FILL_HEALTHY, FILL_CANCER]
    labels = ["Healthy", "Lung cancer"]

    ax.bar(
        x,
        heights,
        width=0.55,
        color=colors,
        edgecolor="black",
        linewidth=BAR_LW,
        alpha=0.92,
        zorder=2,
        yerr=yerr,
        capsize=10,
        error_kw=dict(elinewidth=2.2, ecolor="black", capthick=2.2),
    )

    _annotate_delta_beta(ax, bar_h, bar_c)

    if len(healthy) >= 1 and len(cancer) >= 1:
        _, p = stats.mannwhitneyu(healthy, cancer, alternative="two-sided")
        ax.text(
            0.5,
            0.97,
            f"Mann–Whitney U, p = {_format_p_exact(p)}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=FONT_ANNOT,
            fontweight="bold",
        )

    ax.set_xlim(-0.45, 1.72)
    ax.set_xticks(x, labels)
    ax.set_ylim(*BETA_YLIM)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=8)
    ax.grid(True, alpha=0.22, axis="y")
    ax.tick_params(axis="both", labelsize=FONT_TICK)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rebuild-beta", action="store_true")
    ap.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    ap.add_argument("--out-pdf", type=Path, default=DEFAULT_OUT_PDF)
    args = ap.parse_args()

    for path in (SL_GENOME_CSV, SL_ZSCORE_CSV, EXT_METH_CSV, H_BETA_DIR, L_BETA_DIR):
        if not path.exists():
            raise SystemExit(f"Missing required path: {path}")

    sl_genome = _load_sl_genomewide()
    sl_windows = {
        r["window_id"]: _load_sl_window_zscore(r["window_id"])
        for r in REGIONS
        if r["window_id"] is not None
    }
    beta_by_region = _load_beta_by_region(args.rebuild_beta)

    plt.rcParams.update(
        {
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "font.size": FONT_TICK,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), sharex=False)
    fig.subplots_adjust(left=0.11, right=0.96, top=0.93, bottom=0.08, wspace=0.42, hspace=0.38)

    for col, region in enumerate(REGIONS):
        wid = region["window_id"]
        title = region["title"]

        if wid is None:
            h_sl, c_sl = sl_genome
            sl_ylim = (0.0, 1.0)
            zscore_line = False
            p_sci_sl = False
        else:
            h_sl, c_sl = sl_windows[wid]
            sl_ylim = None
            zscore_line = True
            p_sci_sl = True

        _draw_boxplot_panel(
            axes[0, col],
            h_sl,
            c_sl,
            seed=args.seed + col,
            title=title,
            ylim=sl_ylim,
            zscore_line=zscore_line,
            p_scientific=p_sci_sl,
        )

        h_b, c_b = beta_by_region[wid]
        _draw_external_beta_bar_panel(
            axes[1, col],
            h_b,
            c_b,
            wid,
            title=title,
        )

    axes[0, 0].set_ylabel("Short/long ratio", fontsize=FONT_LABEL, fontweight="bold")
    axes[0, 1].set_ylabel("Short/long ratio (z-scored)", fontsize=FONT_LABEL, fontweight="bold")
    axes[1, 0].set_ylabel(r"$\beta$ value of external DNA Methyl", fontsize=FONT_LABEL, fontweight="bold")
    axes[1, 1].set_ylabel(r"$\beta$ value of external DNA Methyl", fontsize=FONT_LABEL, fontweight="bold")

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=300, bbox_inches="tight")
    fig.savefig(args.out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {args.out_png}")
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
