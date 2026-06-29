#!/usr/bin/env python3
"""Generate data/supplementary_file_1_coupling_weights.csv (Supplementary File 1)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from mfrag.coupling_weights import build_supplementary_file_1
from mfrag.paths import EXT_METH_LUNG, SUPPLEMENTARY_FILE_1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meth-csv", type=Path, default=EXT_METH_LUNG)
    ap.add_argument("--out-csv", type=Path, default=SUPPLEMENTARY_FILE_1)
    ap.add_argument("--mds-lambda", type=float, default=0.5)
    args = ap.parse_args()
    df = build_supplementary_file_1(args.meth_csv, args.out_csv, mds_lambda=args.mds_lambda)
    print(f"Wrote {args.out_csv} ({len(df)} windows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
