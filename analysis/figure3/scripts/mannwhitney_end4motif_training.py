#!/usr/bin/env python3
"""
Per-sample weighted 5' end 4-mer motif frequencies; Mann-Whitney U vs group.

Each fragment contributes two motif counts (end_4_motif_5p, end_4_motif_5m),
weighted by fragment weight. Per-sample frequency = motif_weight / sample_total * 100.

Output CSV sorted by delta_median = healthy_median - lung_cancer_median.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None

MATHIOS = Path(__file__).resolve().parents[1]
TRAIN_DIR = MATHIOS / "fragments_csv" / "training"
OUT_DIR = Path(__file__).resolve().parent
MOTIFS = ["".join(p) for p in product("ACGT", repeat=4)]
MOTIF_INDEX = {m: i for i, m in enumerate(MOTIFS)}
MOTIF_COLS = ["end_4_motif_5p", "end_4_motif_5m"]
CHUNK = 2_000_000
OUT_CSV = OUT_DIR / "end4motif_training_mannwhitney.csv"


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
    if total <= 0:
        freqs = np.zeros(256, dtype=np.float64)
    else:
        freqs = weights / total * 100.0
    group = "healthy" if path.name.endswith("_healthy.csv") else "lung_cancer"
    sample_id = path.stem
    return sample_id, group, freqs


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    if multipletests is not None:
        _, adj, _, _ = multipletests(p_values, method="fdr_bh")
        return adj
    # Fallback Benjamini-Hochberg
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adj = np.empty(n, dtype=np.float64)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(adj, 0, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    files = sorted(TRAIN_DIR.glob("*_healthy.csv")) + sorted(
        TRAIN_DIR.glob("*_lung_cancer.csv")
    )
    print(f"Training samples: {len(files)}")

    rows: list[tuple[str, str, np.ndarray]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_sample_frequencies, p): p for p in files}
        done = 0
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            if done % 20 == 0 or done == len(files):
                print(f"  processed {done}/{len(files)} samples", flush=True)

    healthy = [freqs for _, grp, freqs in rows if grp == "healthy"]
    cancer = [freqs for _, grp, freqs in rows if grp == "lung_cancer"]
    h_mat = np.vstack(healthy)
    c_mat = np.vstack(cancer)
    print(f"Healthy samples: {h_mat.shape[0]}; lung cancer samples: {c_mat.shape[0]}")

    records = []
    p_values = []
    for i, motif in enumerate(MOTIFS):
        h_vals = h_mat[:, i]
        c_vals = c_mat[:, i]
        h_med = float(np.median(h_vals))
        c_med = float(np.median(c_vals))
        if (len(np.unique(h_vals)) <= 1 and len(np.unique(c_vals)) <= 1
                and np.unique(h_vals)[0] == np.unique(c_vals)[0]):
            p = 1.0
        else:
            res = mannwhitneyu(h_vals, c_vals, alternative="two-sided")
            p = float(res.pvalue)
        records.append(
            {
                "motif": motif,
                "healthy_median_frequency_pct": h_med,
                "lung_cancer_median_frequency_pct": c_med,
                "delta_median": h_med - c_med,
                "p_value": p,
            }
        )
        p_values.append(p)

    adj = _bh_adjust(np.array(p_values, dtype=np.float64))
    for rec, adj_p in zip(records, adj):
        rec["adj_p_value"] = float(adj_p)

    df = pd.DataFrame(records)
    df = df.sort_values("delta_median", ascending=False).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.8f")
    print(f"Wrote {OUT_CSV}")
    print(f"Top delta_median: {df.iloc[0]['motif']} ({df.iloc[0]['delta_median']:.6f})")
    print(f"Bottom delta_median: {df.iloc[-1]['motif']} ({df.iloc[-1]['delta_median']:.6f})")
    sig = (df["adj_p_value"] < 0.05).sum()
    print(f"Motifs with adj_p_value < 0.05: {sig}")


if __name__ == "__main__":
    main()
