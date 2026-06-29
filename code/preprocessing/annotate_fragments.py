#!/usr/bin/env python3
"""CLI: annotate fragment CSV with GC weights, end-4-mer motifs, and reference methylation."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
os.environ.setdefault("MFRAG_ROOT", str(CODE_DIR.parent))

from mfrag.paths import POOLED_HEALTHY_CPG, POOLED_LUNG_CPG
from mfrag.preprocessing.gc_weights import annotate_fragments_gc_weights
from mfrag.preprocessing.motifs import annotate_end4_motifs
from mfrag.preprocessing.reference_methylation import annotate_reference_methylation


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fragments-csv", type=Path, required=True)
    ap.add_argument("--sample-stem", type=str, required=True, help="Column name in GC weights table")
    ap.add_argument("--gc-weights-csv", type=Path, required=True)
    ap.add_argument("--ref-fasta", type=Path, required=True, help="hg19 FASTA for motif annotation")
    ap.add_argument("--healthy-cpg-csv", type=Path, default=POOLED_HEALTHY_CPG)
    ap.add_argument("--lung-cpg-csv", type=Path, default=POOLED_LUNG_CPG)
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output path (default: overwrite --fragments-csv in place)",
    )
    args = ap.parse_args()

    src = args.fragments_csv
    out = args.out_csv or src
    in_place = out.resolve() == src.resolve()
    work = src.with_suffix(".annotated.tmp.csv") if in_place else out

    annotate_fragments_gc_weights(src, args.gc_weights_csv, args.sample_stem, out_csv=work)
    annotate_end4_motifs(work, args.ref_fasta)
    annotate_reference_methylation(work, args.healthy_cpg_csv, args.lung_cpg_csv)

    if in_place:
        work.replace(out)

    print(f"Wrote annotated fragment CSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
