"""Annotate fragment CSVs with 5′ end 4-mer motifs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

MOTIF_5P = "end_4_motif_5p"
MOTIF_5M = "end_4_motif_5m"
_COMP = str.maketrans("ACGTN", "TGCAN")
CHUNK = 500_000


def revcomp(seq: str) -> str:
    return seq.upper().translate(_COMP)[::-1]


def annotate_end4_motifs(
    fragments_csv: Path | str,
    ref_fasta: Path | str,
    *,
    out_csv: Path | str | None = None,
    chunksize: int = CHUNK,
) -> Path:
    """
    Add ``end_4_motif_5p`` and ``end_4_motif_5m`` columns to a fragment CSV.

    Requires columns: chromosome, start, end (1-based inclusive coordinates).
    """
    try:
        import pyfaidx
    except ImportError as exc:
        raise ImportError("pyfaidx is required for motif annotation: pip install pyfaidx") from exc

    src = Path(fragments_csv)
    dst = Path(out_csv) if out_csv else src
    fa = pyfaidx.Fasta(str(ref_fasta))

    def norm_chrom(chrom: str) -> str:
        c = str(chrom)
        if c in fa:
            return c
        if not c.startswith("chr") and f"chr{c}" in fa:
            return f"chr{c}"
        return c

    dst.parent.mkdir(parents=True, exist_ok=True)
    header = True
    for chunk in pd.read_csv(src, chunksize=chunksize):
        chroms = chunk["chromosome"].map(norm_chrom)
        motifs_5p = []
        motifs_5m = []
        for chrom, start, end in zip(chroms, chunk["start"], chunk["end"]):
            s0 = int(start) - 1
            e0 = int(end)
            if chrom not in fa or e0 - s0 < 4:
                motifs_5p.append("")
                motifs_5m.append("")
                continue
            plus_5p = str(fa[chrom][s0 : s0 + 4]).upper()
            plus_3p = str(fa[chrom][e0 - 4 : e0]).upper()
            motifs_5p.append(plus_5p)
            motifs_5m.append(revcomp(plus_3p))
        chunk = chunk.copy()
        chunk[MOTIF_5P] = motifs_5p
        chunk[MOTIF_5M] = motifs_5m
        chunk.to_csv(dst, mode="w" if header else "a", header=header, index=False)
        header = False
    return dst
