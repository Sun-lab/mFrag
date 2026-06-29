#!/usr/bin/env python3
"""CLI: BAM → fully annotated fragment CSV (extract + GC weights + motifs + reference methylation)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
os.environ.setdefault("MFRAG_ROOT", str(CODE_DIR.parent))

from mfrag.paths import POOLED_HEALTHY_CPG, POOLED_LUNG_CPG
from mfrag.preprocessing.pipeline import prepare_fragment_csv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bam", type=Path, required=True)
    ap.add_argument("--ref-fasta", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--sample-stem", type=str, required=True)
    ap.add_argument("--gc-weights-csv", type=Path, required=True)
    ap.add_argument("--healthy-cpg-csv", type=Path, default=POOLED_HEALTHY_CPG)
    ap.add_argument("--lung-cpg-csv", type=Path, default=POOLED_LUNG_CPG)
    args = ap.parse_args()

    out = prepare_fragment_csv(
        args.bam,
        args.ref_fasta,
        args.out_csv,
        sample_stem=args.sample_stem,
        gc_weights_csv=args.gc_weights_csv,
        healthy_cpg_csv=args.healthy_cpg_csv,
        lung_cpg_csv=args.lung_cpg_csv,
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
