#!/usr/bin/env python3
"""Genome-wide bar profiles: per-window motif median frequency difference vs genome-wide."""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

_scripts = Path(__file__).resolve().parents[1] / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from nmb_window_ratios_core import (  # noqa: E402
    CHROM_SIZES,
    assign_bins_vectorized,
    build_chrom_bin_offset,
    create_genomic_bins,
)
from plot_shannon_entropy_by_window import (  # noqa: E402
    _chromosome_boundaries,
    _window_label,
)
from plot_window_boxplots_top4_motifs import _full_window_label  # noqa: E402
from window_mannwhitney_top4_motifs import (
    BIN_SIZE,
    CACHE_NPZ,
    CHUNK,
    MOTIF_COLS,
    TOP4,
    TRAIN_FRAG,
    WINDOW_IDS,
    _load_or_build,
)

HERE = Path(__file__).resolve().parent
CFCRAFT = HERE.parents[1]
FIGURE3_DIR = CFCRAFT / "mFrag_Figures" / "Figure3"
MW_CSV = HERE / "end4motif_training_mannwhitney.csv"
TOP5 = ["CCTG", "CCCA", "CCAG", "CCAA", "CCTT"]
NEG_LOG10_P_MOTIFS = frozenset({"CCTG", "CCCA"})
VOLCANO_MOTIFS = frozenset({"CCTG", "CCCA"})
VOLCANO_XMAX = 0.13
COLOR_POS = "#1f77b4"
COLOR_NEG = "#d62728"
COLOR_TOP_PEAK = "#9467bd"
N_TOP_PEAKS = 2
PEAK_BAR_WIDTH = 1.55
# Only color red when clearly negative (avoids tiny noise bars near chr boundaries).
NEG_COLOR_MIN = 0.01

BINS = create_genomic_bins(BIN_SIZE)
CHROM_OFFSET = build_chrom_bin_offset(BINS)
N_WIN = len(BINS)


def _sample_window_single_motif(path_str: str, motif: str) -> tuple[str, str, np.ndarray]:
    path = Path(path_str)
    motif_w = np.zeros(N_WIN, dtype=np.float64)
    total_w = np.zeros(N_WIN, dtype=np.float64)

    for chunk in pd.read_csv(
        path,
        usecols=["chromosome", "start", "weight", *MOTIF_COLS],
        chunksize=CHUNK,
    ):
        df = chunk.rename(columns={"chromosome": "chrom"})
        df["chrom"] = df["chrom"].astype(str)
        df["chrom"] = df["chrom"].where(
            df["chrom"].str.startswith("chr"), "chr" + df["chrom"]
        )
        df = df[df["chrom"].isin(CHROM_SIZES.keys())]
        if df.empty:
            continue

        gen_bin = assign_bins_vectorized(df, CHROM_OFFSET, BIN_SIZE)
        w = df["weight"].astype(np.float64).values

        for col in MOTIF_COLS:
            s = df[col].astype(str).str.upper().str.strip()
            valid = (
                df[col].notna()
                & s.str.len().eq(4)
                & s.str.match(r"^[ACGT]{4}$", na=False)
                & (gen_bin >= 0)
            )
            if not valid.any():
                continue
            gb = gen_bin[valid]
            wt = w[valid]
            motifs = s[valid].values
            np.add.at(total_w, gb, wt)
            mask = motifs == motif
            if mask.any():
                np.add.at(motif_w, gb[mask], wt[mask])

    freqs = np.zeros(N_WIN, dtype=np.float64)
    nz = total_w > 0
    freqs[nz] = motif_w[nz] / total_w[nz] * 100.0
    group = "healthy" if path.name.endswith("_healthy.csv") else "lung_cancer"
    return path.stem, group, freqs


def _load_single_motif_cache(motif: str, workers: int) -> tuple[np.ndarray, np.ndarray]:
    cache = HERE / f"end4motif_{motif}_per_window_frequencies.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        print(f"Loaded cache {cache}")
        return data["healthy_freqs"], data["cancer_freqs"]

    files = sorted(TRAIN_FRAG.glob("*_healthy.csv")) + sorted(
        TRAIN_FRAG.glob("*_lung_cancer.csv")
    )
    print(f"Building per-window {motif} frequencies from {len(files)} samples ...")
    healthy, cancer = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_sample_window_single_motif, str(p), motif): p for p in files}
        done = 0
        for fut in as_completed(futs):
            _, group, freqs = fut.result()
            (healthy if group == "healthy" else cancer).append(freqs)
            done += 1
            if done % 20 == 0 or done == len(files):
                print(f"  {motif}: processed {done}/{len(files)} samples", flush=True)

    h_arr = np.stack(healthy, axis=0)
    c_arr = np.stack(cancer, axis=0)
    np.savez(cache, healthy_freqs=h_arr, cancer_freqs=c_arr)
    print(f"Wrote cache {cache}")
    return h_arr, c_arr


def _load_motif_arrays(
    motif: str, workers: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (n_samples, n_windows) healthy/cancer frequency arrays for one motif."""
    if motif in TOP4:
        h_all, c_all = _load_or_build(workers)
        j = TOP4.index(motif)
        return h_all[:, :, j], c_all[:, :, j]
    return _load_single_motif_cache(motif, workers)


def _genomewide_delta_median(motif: str) -> float:
    df = pd.read_csv(MW_CSV)
    row = df.loc[df["motif"] == motif]
    if row.empty:
        raise SystemExit(f"{motif} not found in {MW_CSV}")
    return float(row.iloc[0]["delta_median"])


def _per_window_delta_median(h_arr: np.ndarray, c_arr: np.ndarray) -> np.ndarray:
    return np.median(h_arr, axis=0) - np.median(c_arr, axis=0)


def _per_window_pvalues(h_arr: np.ndarray, c_arr: np.ndarray) -> np.ndarray:
    """Two-sided Mann-Whitney U p-value per window (healthy vs lung cancer)."""
    n_win = h_arr.shape[1]
    pvals = np.full(n_win, np.nan, dtype=np.float64)
    for w in range(n_win):
        h_vals = h_arr[:, w]
        c_vals = c_arr[:, w]
        if not ((h_vals > 0).all() and (c_vals > 0).all()):
            continue
        if (
            len(np.unique(h_vals)) <= 1
            and len(np.unique(c_vals)) <= 1
            and np.unique(h_vals)[0] == np.unique(c_vals)[0]
        ):
            pvals[w] = 1.0
        else:
            pvals[w] = float(mannwhitneyu(h_vals, c_vals, alternative="two-sided").pvalue)
    return pvals


def _neg_log10_p(pvals: np.ndarray) -> np.ndarray:
    out = np.full_like(pvals, np.nan, dtype=np.float64)
    ok = np.isfinite(pvals) & (pvals > 0)
    out[ok] = -np.log10(np.clip(pvals[ok], 1e-300, 1.0))
    return out


def _genomewide_pvalue(motif: str) -> float:
    df = pd.read_csv(MW_CSV)
    row = df.loc[df["motif"] == motif]
    if row.empty:
        raise SystemExit(f"{motif} not found in {MW_CSV}")
    return float(row.iloc[0]["p_value"])


def _style_genome_axis(ax, window_ids: list[str]) -> None:
    labels = [_window_label(w) for w in window_ids]
    boundaries = _chromosome_boundaries(window_ids)
    n = len(window_ids)

    for i, (start, _) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else n
        if i % 2 == 0:
            ax.axvspan(start - 0.5, end - 0.5, color="0.96", zorder=0)

    for chrom_i, (xi, _) in enumerate(boundaries):
        if chrom_i > 0:
            ax.axvline(xi - 0.5, color="#888888", linestyle=":", linewidth=0.6, alpha=0.5)

    tick_pos = [m[0] for m in boundaries]
    tick_lbl = [labels[i] for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=45, ha="right", fontsize=7, fontweight="bold")
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    ax.set_xlim(-0.5, n - 0.5)


def _plot_motif(
    motif: str,
    h_arr: np.ndarray,
    c_arr: np.ndarray,
    *,
    out_png: Path,
    out_pdf: Path,
    out_csv: Path,
) -> None:
    win_delta = _per_window_delta_median(h_arr, c_arr)
    gw_delta = _genomewide_delta_median(motif)

    pd.DataFrame(
        {
            "window_id": WINDOW_IDS,
            "delta_median_pct": win_delta,
            "deviation_from_genomewide": win_delta - gw_delta,
        }
    ).to_csv(out_csv, index=False, float_format="%.8f")
    print(f"Wrote {out_csv}")
    print(f"Genome-wide median difference ({motif}): {gw_delta:.4f}%")

    x = np.arange(len(WINDOW_IDS))
    colors = np.where(win_delta < -NEG_COLOR_MIN, COLOR_NEG, COLOR_POS)
    top_idx = np.argsort(np.abs(win_delta))[-N_TOP_PEAKS:]
    for i in top_idx:
        print(
            f"  Highlight peak {i + 1}/{N_TOP_PEAKS}: "
            f"{WINDOW_IDS[i]} (Δ={win_delta[i]:.4f}%)"
        )

    fig, ax = plt.subplots(figsize=(18, 5))
    _style_genome_axis(ax, WINDOW_IDS)
    ax.bar(x, win_delta, width=1.0, color=colors, alpha=0.82, edgecolor="none", zorder=3)
    ax.bar(
        x[top_idx],
        win_delta[top_idx],
        width=PEAK_BAR_WIDTH,
        color=COLOR_TOP_PEAK,
        alpha=0.92,
        edgecolor="none",
        zorder=4,
    )
    ax.axhline(
        gw_delta,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=f"Genome-wide median $\\Delta$ = {gw_delta:.3f}%",
        zorder=5,
    )
    ax.axhline(0.0, color="0.5", linestyle=":", linewidth=0.8, zorder=1)
    ax.set_ylabel(
        f"{motif} median frequency difference\n(healthy $-$ lung cancer, %)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(
        f"{motif} per-window vs genome-wide median frequency difference",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right", framealpha=0.9, prop={"size": 10, "weight": "bold"})
    ax.grid(True, axis="y", alpha=0.25)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.18)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def _plot_motif_neg_log10_p(
    motif: str,
    h_arr: np.ndarray,
    c_arr: np.ndarray,
    *,
    out_png: Path,
    out_pdf: Path,
    out_csv: Path,
) -> None:
    win_delta = _per_window_delta_median(h_arr, c_arr)
    pvals = _per_window_pvalues(h_arr, c_arr)
    neg_log_p = _neg_log10_p(pvals)
    gw_p = _genomewide_pvalue(motif)
    gw_neg_log_p = float(-np.log10(max(gw_p, 1e-300)))

    pd.DataFrame(
        {
            "window_id": WINDOW_IDS,
            "delta_median_pct": win_delta,
            "p_value": pvals,
            "neg_log10_p": neg_log_p,
        }
    ).to_csv(out_csv, index=False, float_format="%.8f")
    print(f"Wrote {out_csv}")
    print(f"Genome-wide Mann-Whitney p ({motif}): {gw_p:.4e} (-log10 p = {gw_neg_log_p:.3f})")

    x = np.arange(len(WINDOW_IDS))
    y = np.nan_to_num(neg_log_p, nan=0.0)
    colors = np.where(win_delta < -NEG_COLOR_MIN, COLOR_NEG, COLOR_POS)
    valid = np.isfinite(neg_log_p)
    top_idx = np.argsort(np.where(valid, neg_log_p, -np.inf))[-N_TOP_PEAKS:]
    for i in top_idx:
        print(
            f"  Highlight peak {i + 1}/{N_TOP_PEAKS}: "
            f"{WINDOW_IDS[i]} (p={pvals[i]:.4e}, -log10 p={neg_log_p[i]:.3f})"
        )

    fig, ax = plt.subplots(figsize=(18, 5))
    _style_genome_axis(ax, WINDOW_IDS)
    ax.bar(x, y, width=1.0, color=colors, alpha=0.82, edgecolor="none", zorder=3)
    ax.bar(
        x[top_idx],
        y[top_idx],
        width=PEAK_BAR_WIDTH,
        color=COLOR_TOP_PEAK,
        alpha=0.92,
        edgecolor="none",
        zorder=4,
    )
    ax.axhline(
        gw_neg_log_p,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=f"Genome-wide $-\\log_{{10}}(p)$ = {gw_neg_log_p:.2f}",
        zorder=5,
    )
    ax.set_ylabel(
        f"{motif} $-\\log_{{10}}(p)$\n(Mann-Whitney U, healthy vs lung cancer)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(
        f"{motif} per-window motif frequency significance",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right", framealpha=0.9, prop={"size": 10, "weight": "bold"})
    ax.grid(True, axis="y", alpha=0.25)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.18)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def _plot_motif_volcano(
    motif: str,
    h_arr: np.ndarray,
    c_arr: np.ndarray,
    *,
    out_png: Path,
    out_pdf: Path,
    out_csv: Path,
) -> None:
    win_delta = _per_window_delta_median(h_arr, c_arr)
    pvals = _per_window_pvalues(h_arr, c_arr)
    neg_log_p = _neg_log10_p(pvals)
    gw_p = _genomewide_pvalue(motif)
    gw_neg_log_p = float(-np.log10(max(gw_p, 1e-300)))

    pd.DataFrame(
        {
            "window_id": WINDOW_IDS,
            "delta_median_pct": win_delta,
            "p_value": pvals,
            "neg_log10_p": neg_log_p,
        }
    ).to_csv(out_csv, index=False, float_format="%.8f")
    print(f"Wrote {out_csv}")
    print(f"Genome-wide Mann-Whitney p ({motif}): {gw_p:.4e} (-log10 p = {gw_neg_log_p:.3f})")

    valid = np.isfinite(neg_log_p) & np.isfinite(win_delta)
    x = win_delta[valid]
    y = neg_log_p[valid]
    wids = [WINDOW_IDS[i] for i in np.where(valid)[0]]

    colors = np.where(x < -NEG_COLOR_MIN, COLOR_NEG, COLOR_POS)
    top_idx = np.argsort(np.where(valid, neg_log_p, -np.inf))[-N_TOP_PEAKS:]
    peak_set = {WINDOW_IDS[i] for i in top_idx}
    for i in top_idx:
        print(
            f"  Annotate peak {i + 1}/{N_TOP_PEAKS}: "
            f"{WINDOW_IDS[i]} (Δ={win_delta[i]:.4f}%, p={pvals[i]:.4e})"
        )

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    ax.scatter(
        x,
        y,
        c=colors,
        s=28,
        alpha=0.72,
        edgecolors="0.35",
        linewidths=0.25,
        zorder=2,
    )

    peak_mask = np.array([w in peak_set for w in wids])
    if peak_mask.any():
        ax.scatter(
            x[peak_mask],
            y[peak_mask],
            s=90,
            facecolors="none",
            edgecolors=COLOR_TOP_PEAK,
            linewidths=1.6,
            zorder=4,
        )

    y_top = float(np.nanmax(y)) if len(y) else gw_neg_log_p
    ax.set_ylim(0.0, y_top + 0.35)
    ax.set_xlim(float(np.nanmin(x)) - 0.02, VOLCANO_XMAX)

    peaks_sorted = sorted(peak_set, key=lambda w: -neg_log_p[WINDOW_IDS.index(w)])
    peak_x = {wid: float(win_delta[WINDOW_IDS.index(wid)]) for wid in peaks_sorted}
    for rank, wid in enumerate(peaks_sorted):
        idx = WINDOW_IDS.index(wid)
        if not valid[idx]:
            continue
        px = float(win_delta[idx])
        py = float(neg_log_p[idx])
        is_left_peak = len(peaks_sorted) > 1 and px <= min(
            peak_x[w] for w in peaks_sorted if w != wid
        )
        if is_left_peak:
            tx, ty, ha, va = px - 0.006, py - 0.55, "right", "top"
        elif rank == 0:
            tx, ty, ha, va = px + 0.018, py - 0.52, "left", "top"
        else:
            tx, ty, ha, va = px + 0.018, py - 0.42, "left", "top"
        ax.annotate(
            _full_window_label(wid),
            xy=(px, py),
            xytext=(tx, ty),
            textcoords="data",
            fontsize=10,
            fontweight="bold",
            ha=ha,
            va=va,
            clip_on=True,
            arrowprops=dict(
                arrowstyle="-",
                color=COLOR_TOP_PEAK,
                lw=0.8,
                alpha=0.85,
                shrinkA=2,
                shrinkB=2,
            ),
            bbox=dict(
                boxstyle="round,pad=0.15",
                fc="white",
                ec="0.7",
                lw=0.3,
                alpha=0.92,
            ),
            zorder=5,
        )

    ax.axhline(
        gw_neg_log_p,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label=f"Genome-wide $-\\log_{{10}}(p)$ = {gw_neg_log_p:.2f}",
        zorder=1,
    )
    ax.axvline(0.0, color="0.5", linestyle=":", linewidth=0.8, zorder=1)
    ax.set_xlabel(
        f"{motif} median frequency difference (healthy $-$ lung cancer, %)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel(
        f"$-\\log_{{10}}(p)$ (Mann-Whitney U, healthy vs lung cancer)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper right", framealpha=0.9, prop={"size": 10, "weight": "bold"})
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--motifs",
        nargs="+",
        default=TOP5,
        choices=TOP5,
        help=f"Motifs to plot (default: all top-5: {TOP5})",
    )
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--cache", type=Path, default=CACHE_NPZ)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
        }
    )

    h_top4, c_top4 = None, None
    for motif in args.motifs:
        if motif in TOP4:
            if h_top4 is None:
                if args.cache.exists():
                    data = np.load(args.cache, allow_pickle=True)
                    h_top4, c_top4 = data["healthy_freqs"], data["cancer_freqs"]
                    print(f"Loaded cache {args.cache}")
                else:
                    h_top4, c_top4 = _load_or_build(args.workers)
            j = TOP4.index(motif)
            h_arr, c_arr = h_top4[:, :, j], c_top4[:, :, j]
        else:
            h_arr, c_arr = _load_motif_arrays(motif, args.workers)

        FIGURE3_DIR.mkdir(parents=True, exist_ok=True)
        out_png = FIGURE3_DIR / f"end4motif_{motif}_window_median_difference.png"
        out_pdf = FIGURE3_DIR / f"end4motif_{motif}_window_median_difference.pdf"
        out_csv = HERE / f"end4motif_{motif}_window_median_difference.csv"
        if motif in VOLCANO_MOTIFS:
            _plot_motif_volcano(
                motif, h_arr, c_arr, out_png=out_png, out_pdf=out_pdf, out_csv=out_csv
            )
        elif motif in NEG_LOG10_P_MOTIFS:
            _plot_motif_neg_log10_p(
                motif, h_arr, c_arr, out_png=out_png, out_pdf=out_pdf, out_csv=out_csv
            )
        else:
            _plot_motif(motif, h_arr, c_arr, out_png=out_png, out_pdf=out_pdf, out_csv=out_csv)


if __name__ == "__main__":
    main()
