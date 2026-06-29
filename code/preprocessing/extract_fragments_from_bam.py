#!/usr/bin/env python3
"""CLI: BAM → fragment CSV (chromosome, start, end, fragment_length, gc_content)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
os.environ.setdefault("MFRAG_ROOT", str(CODE_DIR.parent))

from mfrag.preprocessing.bam import extract_fragments_from_bam


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bam", type=Path, required=True, help="Input aligned BAM")
    ap.add_argument("--ref-fasta", type=Path, required=True, help="hg19 reference FASTA (.fai required)")
    ap.add_argument("--out-csv", type=Path, required=True, help="Output fragment CSV")
    ap.add_argument("--max-fragments", type=int, default=None, help="Optional cap for testing")
    args = ap.parse_args()
    n = extract_fragments_from_bam(args.bam, args.ref_fasta, args.out_csv, max_fragments=args.max_fragments)
    print(f"Wrote {args.out_csv} ({n:,} fragments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
