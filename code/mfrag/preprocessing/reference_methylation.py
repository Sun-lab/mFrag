"""Map external WGBS reference methylation to fragment intervals."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from mfrag.preprocessing.feature_bins import COL_HEALTHY_METH, COL_LUNG_METH

CHUNK = 500_000


def load_pooled_cpg_index(
    cpg_csv: Path | str,
    beta_column: str = "beta_value",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load pooled single-CpG table into per-chromosome sorted position + prefix-sum arrays."""
    buckets: dict[str, list] = defaultdict(lambda: [[], []])
    for chunk in pd.read_csv(cpg_csv, usecols=["chrom", "position", beta_column], chunksize=2_000_000):
        for chrom, grp in chunk.groupby("chrom", sort=False):
            c = str(chrom)
            buckets[c][0].append(grp["position"].to_numpy(dtype=np.int32, copy=False))
            buckets[c][1].append(grp[beta_column].to_numpy(dtype=np.float64, copy=False))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chrom, (pos_parts, beta_parts) in buckets.items():
        if not pos_parts:
            continue
        pos = np.concatenate(pos_parts)
        beta = np.concatenate(beta_parts)
        order = np.argsort(pos, kind="mergesort")
        pos = pos[order]
        beta = beta[order]
        psum = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(beta)))
        out[chrom] = (pos, psum)
    return out


def _normalize_chrom_key(chrom: str, index: dict) -> str:
    c = str(chrom)
    if c in index:
        return c
    if c.startswith("chr") and c[3:] in index:
        return c[3:]
    if not c.startswith("chr") and ("chr" + c) in index:
        return "chr" + c
    return c


def interval_mean_beta(
    chrom: str,
    starts: np.ndarray,
    ends: np.ndarray,
    index: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Mean β over CpGs with start <= position <= end (1-based inclusive fragment coords)."""
    out = np.full(len(starts), np.nan, dtype=np.float64)
    key = _normalize_chrom_key(chrom, index)
    if key not in index:
        return out
    pos, psum = index[key]
    for i, (s, e) in enumerate(zip(starts, ends)):
        lo = int(s)
        hi = int(e)
        left = np.searchsorted(pos, lo, side="left")
        right = np.searchsorted(pos, hi, side="right")
        n = right - left
        if n <= 0:
            continue
        out[i] = (psum[right] - psum[left]) / n
    return out


def annotate_reference_methylation(
    fragments_csv: Path | str,
    healthy_cpg_csv: Path | str,
    lung_cpg_csv: Path | str,
    *,
    out_csv: Path | str | None = None,
    chunksize: int = CHUNK,
) -> Path:
    """Add HealthyRef_DNAMethylation and LungcancerRef_DNAMethylation columns."""
    src = Path(fragments_csv)
    dst = Path(out_csv) if out_csv else src
    idx_h = load_pooled_cpg_index(healthy_cpg_csv)
    idx_l = load_pooled_cpg_index(lung_cpg_csv)

    dst.parent.mkdir(parents=True, exist_ok=True)
    header = True
    for chunk in pd.read_csv(src, chunksize=chunksize):
        chroms = chunk["chromosome"].astype(str)
        starts = chunk["start"].to_numpy(dtype=np.int64)
        ends = chunk["end"].to_numpy(dtype=np.int64)
        h_vals = np.full(len(chunk), np.nan)
        l_vals = np.full(len(chunk), np.nan)
        for chrom in chroms.unique():
            mask = chroms == chrom
            h_vals[mask] = interval_mean_beta(chrom, starts[mask], ends[mask], idx_h)
            l_vals[mask] = interval_mean_beta(chrom, starts[mask], ends[mask], idx_l)
        chunk = chunk.copy()
        chunk[COL_HEALTHY_METH] = h_vals
        chunk[COL_LUNG_METH] = l_vals
        chunk.to_csv(dst, mode="w" if header else "a", header=header, index=False)
        header = False
    return dst
