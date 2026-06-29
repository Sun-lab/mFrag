"""Genomic window definitions for mFrag (589 × 5 Mb bins on hg19 autosomes)."""
from __future__ import annotations

import numpy as np
import pandas as pd

BIN_SIZE = 5_000_000
N_WINDOWS = 589

CHROM_SIZES = {
    "chr1": 249250621,
    "chr2": 243199373,
    "chr3": 198022430,
    "chr4": 191154276,
    "chr5": 180915260,
    "chr6": 171115067,
    "chr7": 159138663,
    "chr8": 146364022,
    "chr9": 141213431,
    "chr10": 135534747,
    "chr11": 135006516,
    "chr12": 133851895,
    "chr13": 115169878,
    "chr14": 107349540,
    "chr15": 102531392,
    "chr16": 90354753,
    "chr17": 81195210,
    "chr18": 78077248,
    "chr19": 59128983,
    "chr20": 63025520,
    "chr21": 48129895,
    "chr22": 51304566,
}


def create_genomic_bins(bin_size: int = BIN_SIZE) -> list[tuple[str, int, int, str]]:
    bins: list[tuple[str, int, int, str]] = []
    for chrom in [f"chr{i}" for i in range(1, 23)]:
        size = CHROM_SIZES[chrom]
        for start in range(0, size, bin_size):
            end = min(start + bin_size, size)
            bins.append((chrom, start, end, f"{chrom}_{start}_{end}"))
    return bins


def build_chrom_bin_offset(bins: list[tuple[str, int, int, str]]) -> dict[str, tuple[int, int]]:
    offset: dict[str, tuple[int, int]] = {}
    idx = 0
    for chrom in [f"chr{i}" for i in range(1, 23)]:
        n = sum(1 for b in bins if b[0] == chrom)
        offset[chrom] = (idx, n)
        idx += n
    return offset


def assign_bins_vectorized(
    df: pd.DataFrame, chrom_offset: dict[str, tuple[int, int]], bin_size: int = BIN_SIZE
) -> np.ndarray:
    """Assign each fragment row to a global window index (0..588)."""
    bin_idx = np.full(len(df), -1, dtype=int)
    chrom_col = "chrom" if "chrom" in df.columns else "chromosome"
    for chrom, (start_idx, n_bins) in chrom_offset.items():
        mask = df[chrom_col].values == chrom
        if not mask.any():
            continue
        pos = df.loc[mask, "start"].values
        local = np.minimum(pos // bin_size, n_bins - 1).astype(int)
        bin_idx[mask] = start_idx + local
    return bin_idx


def normalize_chrom(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    return s.where(s.str.startswith("chr"), "chr" + s)
