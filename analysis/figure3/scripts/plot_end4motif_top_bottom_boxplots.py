#!/usr/bin/env python3
"""Boxplots for top 5 significant motifs (adj_p < 0.05) by delta_median."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

MATHIOS = Path(__file__).resolve().parents[1]
CFCRAFT = MATHIOS.parent
FIGURE3_DIR = CFCRAFT / "mFrag_Figures" / "Figure3"
TRAIN_DIR = MATHIOS / "fragments_csv" / "training"
OUT_DIR = Path(__file__).resolve().parent
MW_CSV = OUT_DIR / "end4motif_training_mannwhitney.csv"
CACHE_NPZ = OUT_DIR / "end4motif_training_per_sample_frequencies.npz"
OUT_PNG = FIGURE3_DIR / "end4motif_training_top5_boxplots.png"
OUT_PNG_LEGACY = OUT_DIR / "end4motif_training_top4_bottom4_boxplots.png"
MOTIFS = ["".join(p) for p in product("ACGT", repeat=4)]
MOTIF_INDEX = {m: i for i, m in enumerate(MOTIFS)}
MOTIF_COLS = ["end_4_motif_5p", "end_4_motif_5m"]
CHUNK = 2_000_000
N_TOP = 2


def _accumulate_motif_weights(
    motifs: pd.Series, weights: pd.Series, arr: np.ndarray
) -> None:
    s = motifs.astype(str).str.upper().str.strip()
    valid = motifs.notna() & s.str.len().eq(4) & s.str.match(r"^[ACGT]{4}$", na=False)
    if not valid.any():
        return
    grouped = (
        pd.DataFrame({"motif": s[valid].values, "weight": weights[valid].values})
        .groupby("motif", sort=False)["weight"]
        .sum()
    )
    for motif, wt in grouped.items():
        idx = MOTIF_INDEX.get(motif)
        if idx is not None:
            arr[idx] += wt


def _sample_weight_vector(path_str: str) -> np.ndarray:
    arr = np.zeros(256, dtype=np.float64)
    for chunk in pd.read_csv(
        Path(path_str), usecols=["weight", *MOTIF_COLS], chunksize=CHUNK
    ):
        w = chunk["weight"].astype(np.float64)
        for col in MOTIF_COLS:
            _accumulate_motif_weights(chunk[col], w, arr)
    return arr


def _sample_frequencies(path: Path) -> tuple[str, str, np.ndarray]:
    weights = _sample_weight_vector(str(path))
    total = weights.sum()
    freqs = np.zeros(256, dtype=np.float64) if total <= 0 else weights / total * 100.0
    group = "healthy" if path.name.endswith("_healthy.csv") else "lung_cancer"
    return path.stem, group, freqs


def _load_or_build_sample_matrix(workers: int) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    if CACHE_NPZ.exists():
        data = np.load(CACHE_NPZ, allow_pickle=True)
        print(f"Loaded cache {CACHE_NPZ}")
        return (
            data["healthy_freqs"],
            data["cancer_freqs"],
            list(data["healthy_samples"]),
            list(data["cancer_samples"]),
        )

    files = sorted(TRAIN_DIR.glob("*_healthy.csv")) + sorted(
        TRAIN_DIR.glob("*_lung_cancer.csv")
    )
    print(f"Building per-sample frequencies from {len(files)} files ...")
    healthy_rows, cancer_rows = [], []
    healthy_ids, cancer_ids = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_sample_frequencies, p): p for p in files}
        done = 0
        for fut in as_completed(futs):
            sid, group, freqs = fut.result()
            if group == "healthy":
                healthy_rows.append(freqs)
                healthy_ids.append(sid)
            else:
                cancer_rows.append(freqs)
                cancer_ids.append(sid)
            done += 1
            if done % 20 == 0 or done == len(files):
                print(f"  processed {done}/{len(files)} samples", flush=True)

    h_mat = np.vstack(healthy_rows)
    c_mat = np.vstack(cancer_rows)
    np.savez(
        CACHE_NPZ,
        healthy_freqs=h_mat,
        cancer_freqs=c_mat,
        healthy_samples=np.array(healthy_ids, dtype=object),
        cancer_samples=np.array(cancer_ids, dtype=object),
    )
    print(f"Wrote cache {CACHE_NPZ}")
    return h_mat, c_mat, healthy_ids, cancer_ids


def _select_motifs() -> tuple[list[str], list[float]]:
    df = pd.read_csv(MW_CSV)
    sig = df[df["adj_p_value"] < 0.05]
    top = sig.nlargest(N_TOP, "delta_median")
    return top["motif"].tolist(), top["p_value"].tolist()


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _p_text(p: float) -> str:
    return f"$p$ = {p:.2e}\n{_sig_stars(p)}"


def _draw_bracket(ax, x1: float, x2: float, y: float, h: float, label: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.4, clip_on=False)
    ax.text(
        (x1 + x2) / 2,
        y + h,
        label,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        clip_on=False,
    )


def _plot_boxplots(
    motifs: list[str],
    p_values: list[float],
    h_mat: np.ndarray,
    c_mat: np.ndarray,
) -> None:
    rng = np.random.default_rng(42)
    n_h, n_c = h_mat.shape[0], c_mat.shape[0]
    n = len(motifs)
    group_gap = 3.0
    box_w = 0.7
    jitter = 0.08

    fig, ax = plt.subplots(figsize=(max(12, n * 1.6), 6))
    centers = []

    for i, (motif, pval) in enumerate(zip(motifs, p_values)):
        idx = MOTIF_INDEX[motif]
        h_vals = h_mat[:, idx]
        c_vals = c_mat[:, idx]
        pos_h = i * group_gap
        pos_c = pos_h + 1.0
        centers.append(pos_h + 0.5)

        bp_h = ax.boxplot(
            h_vals,
            positions=[pos_h],
            widths=box_w,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
            boxprops={"facecolor": "#1f77b4", "edgecolor": "black", "linewidth": 1.2},
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
        )
        bp_c = ax.boxplot(
            c_vals,
            positions=[pos_c],
            widths=box_w,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
            boxprops={"facecolor": "#d62728", "edgecolor": "black", "linewidth": 1.2},
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
        )
        del bp_h, bp_c

        ax.scatter(
            pos_h + rng.normal(0, jitter, len(h_vals)),
            h_vals,
            color="#1f77b4",
            edgecolors="black",
            linewidths=0.3,
            s=14,
            alpha=0.65,
            zorder=3,
        )
        ax.scatter(
            pos_c + rng.normal(0, jitter, len(c_vals)),
            c_vals,
            color="#d62728",
            edgecolors="black",
            linewidths=0.3,
            s=14,
            alpha=0.65,
            zorder=3,
        )

        local_max = float(max(h_vals.max(), c_vals.max()))
        bracket_y = min(local_max + 0.12, 2.55)
        _draw_bracket(ax, pos_h, pos_c, bracket_y, 0.06, _p_text(pval))

    ax.set_xticks(centers)
    ax.set_xticklabels(motifs, fontsize=11, fontweight="bold")
    ax.set_ylabel("5' end 4-mer motif frequency (%)", fontsize=12, fontweight="bold")
    ax.set_ylim(1.0, 3.0)
    ax.tick_params(axis="y", labelsize=11)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    legend_handles = [
        Patch(
            facecolor="#1f77b4",
            edgecolor="black",
            label=f"Healthy (N={n_h})",
        ),
        Patch(
            facecolor="#d62728",
            edgecolor="black",
            label=f"Lung cancer (N={n_c})",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right", prop={"size": 11, "weight": "bold"})
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    FIGURE3_DIR.mkdir(parents=True, exist_ok=True)
    for out_path in (OUT_PNG, OUT_PNG_LEGACY):
        fig.savefig(out_path, dpi=150)
        print(f"Wrote {out_path}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    motifs, p_values = _select_motifs()
    print(f"Selected motifs: {motifs}")
    h_mat, c_mat, _, _ = _load_or_build_sample_matrix(args.workers)
    _plot_boxplots(motifs, p_values, h_mat, c_mat)


if __name__ == "__main__":
    main()
