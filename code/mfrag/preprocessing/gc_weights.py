"""GC-stratum weight assignment for fragment CSVs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_GC_STRATA = 100
CHUNK = 500_000


def gc_strata(gc: np.ndarray) -> np.ndarray:
    return np.clip(gc.astype(np.float64) * N_GC_STRATA, 0, N_GC_STRATA - 1).astype(np.int64)


def load_sample_gc_weights(weights_csv: Path, sample_stem: str) -> np.ndarray:
    """Load per-stratum weights for one sample from a wide weights table."""
    wdf = pd.read_csv(weights_csv, index_col=0)
    wdf.index = wdf.index.astype(int)
    if sample_stem not in wdf.columns:
        raise KeyError(f"{sample_stem} not in {weights_csv}")
    arr = wdf[sample_stem].to_numpy(dtype=np.float64)
    if len(arr) < N_GC_STRATA:
        raise ValueError(f"Expected >={N_GC_STRATA} strata rows in {weights_csv}")
    return arr[:N_GC_STRATA]


def annotate_fragments_gc_weights(
    fragments_csv: Path | str,
    weights_csv: Path | str,
    sample_stem: str,
    *,
    out_csv: Path | str | None = None,
    chunksize: int = CHUNK,
) -> Path:
    """Add a ``weight`` column from GC stratum lookup (in-place if out_csv is None)."""
    src = Path(fragments_csv)
    dst = Path(out_csv) if out_csv else src
    warr = load_sample_gc_weights(Path(weights_csv), sample_stem)

    dst.parent.mkdir(parents=True, exist_ok=True)
    header = True
    for chunk in pd.read_csv(src, chunksize=chunksize):
        if "gc_content" not in chunk.columns:
            raise ValueError(f"{src}: missing gc_content column")
        s = gc_strata(chunk["gc_content"].to_numpy())
        chunk = chunk.copy()
        chunk["weight"] = warr[s]
        chunk.to_csv(dst, mode="w" if header else "a", header=header, index=False)
        header = False
    return dst
