#!/usr/bin/env python3
"""
Pool training healthy vs lung_cancer end 4-mer motifs with fragment weights.

Each fragment contributes two motif counts (end_4_motif_5p and end_4_motif_5m),
each weighted by the fragment's weight column.

Outputs:
  - end4motif_training_weighted_counts.csv
  - end4motif_training_stacked_barplot.png
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MATHIOS = Path(__file__).resolve().parents[1]
TRAIN_DIR = MATHIOS / "fragments_csv" / "training"
OUT_DIR = Path(__file__).resolve().parent
MOTIFS = ["".join(p) for p in product("ACGT", repeat=4)]
MOTIF_INDEX = {m: i for i, m in enumerate(MOTIFS)}
MOTIF_COLS = ["end_4_motif_5p", "end_4_motif_5m"]
CHUNK = 2_000_000


def _accumulate_motif_weights(
    motifs: pd.Series, weights: pd.Series, arr: np.ndarray
) -> float:
    """Add weighted motif counts into arr; return skipped weight."""
    s = motifs.astype(str).str.upper().str.strip()
    valid = motifs.notna() & s.str.len().eq(4) & s.str.match(r"^[ACGT]{4}$", na=False)
    skipped = float(weights[~valid].sum())
    if not valid.any():
        return skipped
    grouped = (
        pd.DataFrame({"motif": s[valid].values, "weight": weights[valid].values})
        .groupby("motif", sort=False)["weight"]
        .sum()
    )
    for motif, wt in grouped.items():
        idx = MOTIF_INDEX.get(motif)
        if idx is not None:
            arr[idx] += wt
    return skipped


def _weight_file(path_str: str) -> tuple[np.ndarray, float]:
    """Return (256-dim weight sums, total invalid/skipped weight)."""
    arr = np.zeros(256, dtype=np.float64)
    skipped = 0.0
    path = Path(path_str)
    for chunk in pd.read_csv(
        path, usecols=["weight", *MOTIF_COLS], chunksize=CHUNK
    ):
        w = chunk["weight"].astype(np.float64)
        for col in MOTIF_COLS:
            skipped += _accumulate_motif_weights(chunk[col], w, arr)
    return arr, skipped


def _pool_weights(paths: list[Path], label: str, workers: int) -> tuple[np.ndarray, float]:
    total = np.zeros(256, dtype=np.float64)
    skipped = 0.0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_weight_file, str(p)): p for p in paths}
        done = 0
        for fut in as_completed(futs):
            arr, sk = fut.result()
            total += arr
            skipped += sk
            done += 1
            if done % 20 == 0 or done == len(paths):
                print(f"  {label}: {done}/{len(paths)} files", flush=True)
    return total, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    healthy_files = sorted(TRAIN_DIR.glob("*_healthy.csv"))
    cancer_files = sorted(TRAIN_DIR.glob("*_lung_cancer.csv"))
    print(f"Healthy training files: {len(healthy_files)}")
    print(f"Lung cancer training files: {len(cancer_files)}")

    print("Pooling healthy weights ...")
    h_arr, h_skipped = _pool_weights(healthy_files, "healthy", args.workers)
    print("Pooling lung cancer weights ...")
    c_arr, c_skipped = _pool_weights(cancer_files, "cancer", args.workers)

    h_total = float(h_arr.sum())
    c_total = float(c_arr.sum())
    print(f"Healthy pooled motif-weight sum: {h_total:,.2f} (skipped: {h_skipped:,.2f})")
    print(f"Cancer pooled motif-weight sum:  {c_total:,.2f} (skipped: {c_skipped:,.2f})")

    h_pct = h_arr / h_total * 100.0 if h_total else np.zeros(256)
    c_pct = c_arr / c_total * 100.0 if c_total else np.zeros(256)
    total_pct = h_pct + c_pct

    df = pd.DataFrame(
        {
            "motif": MOTIFS,
            "healthy_weight_sum": h_arr,
            "healthy_frequency_pct": h_pct,
            "lung_cancer_weight_sum": c_arr,
            "lung_cancer_frequency_pct": c_pct,
            "total_frequency_pct": total_pct,
        }
    )
    df = df.sort_values("total_frequency_pct", ascending=False).reset_index(drop=True)

    counts_csv = OUT_DIR / "end4motif_training_weighted_counts.csv"
    df.to_csv(counts_csv, index=False, float_format="%.6f")
    print(f"Wrote {counts_csv}")

    # Horizontal stacked barplot: y = motifs (highest total freq at top), x = frequency %
    n = len(df)
    y_pos = np.arange(n)
    fig_h = max(12, n * 0.08)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    ax.barh(
        y_pos,
        df["lung_cancer_frequency_pct"],
        color="#d62728",
        label="Lung cancer training samples (N=129)",
        height=0.8,
    )
    ax.barh(
        y_pos,
        df["healthy_frequency_pct"],
        left=df["lung_cancer_frequency_pct"],
        color="#1f77b4",
        label="Healthy training samples (N=158)",
        height=0.8,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["motif"], fontsize=6, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Fragment weight frequency (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("5' End 4-mer motif", fontsize=12, fontweight="bold")
    ax.set_title("Pooled 5' end 4-mer motif weight frequency", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", labelsize=11)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    ax.legend(loc="lower right", prop={"size": 11, "weight": "bold"})
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    plot_png = OUT_DIR / "end4motif_training_stacked_barplot.png"
    fig.savefig(plot_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {plot_png}")


if __name__ == "__main__":
    main()
