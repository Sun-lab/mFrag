#!/usr/bin/env python3
"""
Per-window Mann-Whitney U for top-4 motifs (CCTG, CCCA, CCAG, CCAA).

Uses the same 589 x 5 Mb windows as CNN_need_input_files/training_zscored.
Per-sample per-window weighted motif frequency from fragments_csv/training.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None

_scripts = Path(__file__).resolve().parents[1] / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from nmb_window_ratios_core import (  # noqa: E402
    CHROM_SIZES,
    assign_bins_vectorized,
    build_chrom_bin_offset,
    create_genomic_bins,
)

MATHIOS = Path(__file__).resolve().parents[1]
TRAIN_FRAG = MATHIOS / "fragments_csv" / "training"
REF_ZSCORED = (
    MATHIOS / "CNN_need_input_files" / "training_zscored" / "CGPLLU272P_healthy.csv"
)
OUT_DIR = Path(__file__).resolve().parent
CACHE_NPZ = OUT_DIR / "end4motif_top4_per_window_frequencies.npz"
TOP4 = ["CCTG", "CCCA", "CCAG", "CCAA"]
MOTIF_COLS = ["end_4_motif_5p", "end_4_motif_5m"]
BIN_SIZE = 5_000_000
CHUNK = 2_000_000

BINS = create_genomic_bins(BIN_SIZE)
CHROM_OFFSET = build_chrom_bin_offset(BINS)
N_WIN = len(BINS)
WINDOW_IDS = [b[3] for b in BINS]


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    if multipletests is not None:
        _, adj, _, _ = multipletests(p_values, method="fdr_bh")
        return adj
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adj = np.empty(n, dtype=np.float64)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        val = ranked[i] * n / (i + 1)
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(adj, 0, 1)
    return out


def _sample_window_motif_freqs(path_str: str) -> tuple[str, str, np.ndarray]:
    path = Path(path_str)
    motif_w = np.zeros((N_WIN, len(TOP4)), dtype=np.float64)
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
            for j, motif in enumerate(TOP4):
                mask = motifs == motif
                if mask.any():
                    np.add.at(motif_w[:, j], gb[mask], wt[mask])

    freqs = np.zeros((N_WIN, len(TOP4)), dtype=np.float64)
    nz = total_w > 0
    freqs[nz] = motif_w[nz] / total_w[nz, None] * 100.0

    group = "healthy" if path.name.endswith("_healthy.csv") else "lung_cancer"
    return path.stem, group, freqs


def _load_or_build(workers: int) -> tuple[np.ndarray, np.ndarray]:
    if CACHE_NPZ.exists():
        data = np.load(CACHE_NPZ, allow_pickle=True)
        print(f"Loaded cache {CACHE_NPZ}")
        return data["healthy_freqs"], data["cancer_freqs"]

    files = sorted(TRAIN_FRAG.glob("*_healthy.csv")) + sorted(
        TRAIN_FRAG.glob("*_lung_cancer.csv")
    )
    print(f"Building per-window frequencies from {len(files)} samples ...")
    healthy, cancer = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_sample_window_motif_freqs, str(p)): p for p in files}
        done = 0
        for fut in as_completed(futs):
            _, group, freqs = fut.result()
            (healthy if group == "healthy" else cancer).append(freqs)
            done += 1
            if done % 20 == 0 or done == len(files):
                print(f"  processed {done}/{len(files)} samples", flush=True)

    h_arr = np.stack(healthy, axis=0)
    c_arr = np.stack(cancer, axis=0)
    np.savez(CACHE_NPZ, healthy_freqs=h_arr, cancer_freqs=c_arr)
    print(f"Wrote cache {CACHE_NPZ}")
    return h_arr, c_arr


def _test_motif(h_arr: np.ndarray, c_arr: np.ndarray, motif_idx: int) -> pd.DataFrame:
    records = []
    p_values: list[float] = []
    eligible_mask: list[bool] = []

    for w in range(N_WIN):
        h_vals = h_arr[:, w, motif_idx]
        c_vals = c_arr[:, w, motif_idx]
        h_med = float(np.median(h_vals))
        c_med = float(np.median(c_vals))
        eligible = bool((h_vals > 0).all() and (c_vals > 0).all())
        eligible_mask.append(eligible)

        if not eligible:
            p = np.nan
        elif (
            len(np.unique(h_vals)) <= 1
            and len(np.unique(c_vals)) <= 1
            and np.unique(h_vals)[0] == np.unique(c_vals)[0]
        ):
            p = 1.0
        else:
            p = float(mannwhitneyu(h_vals, c_vals, alternative="two-sided").pvalue)

        records.append(
            {
                "window_id": WINDOW_IDS[w],
                "lung_cancer_median_frequency_pct": c_med,
                "healthy_median_frequency_pct": h_med,
                "delta_median": h_med - c_med,
                "all_samples_nonzero": eligible,
                "p_value": p,
            }
        )
        p_values.append(p)

    adj = np.full(N_WIN, np.nan, dtype=np.float64)
    testable = np.array(eligible_mask, dtype=bool)
    raw_p = np.array(p_values, dtype=np.float64)
    if testable.any():
        adj[testable] = _bh_adjust(raw_p[testable])

    for rec, adj_p in zip(records, adj):
        rec["adj_p_value"] = float(adj_p) if np.isfinite(adj_p) else np.nan
    return pd.DataFrame(records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--stats-only",
        action="store_true",
        help="Recompute stats from cached per-window frequencies only",
    )
    args = ap.parse_args()

    ref = pd.read_csv(REF_ZSCORED, usecols=["window_id"])
    if ref["window_id"].tolist() != WINDOW_IDS:
        raise SystemExit("Window panel mismatch vs training_zscored reference")

    if args.stats_only:
        if not CACHE_NPZ.exists():
            raise SystemExit(f"Missing cache {CACHE_NPZ}")
        data = np.load(CACHE_NPZ, allow_pickle=True)
        h_arr, c_arr = data["healthy_freqs"], data["cancer_freqs"]
    else:
        h_arr, c_arr = _load_or_build(args.workers)
    print(f"Healthy: {h_arr.shape[0]} samples; cancer: {c_arr.shape[0]} samples")

    for j, motif in enumerate(TOP4):
        df = _test_motif(h_arr, c_arr, j)
        out = OUT_DIR / f"end4motif_{motif}_window_mannwhitney.csv"
        df.to_csv(out, index=False, float_format="%.8f")
        eligible = df["all_samples_nonzero"].sum()
        sig = ((df["adj_p_value"] < 0.05) & df["all_samples_nonzero"]).sum()
        print(f"Wrote {out} ({eligible} eligible windows; {sig} with adj_p < 0.05)")


if __name__ == "__main__":
    main()
