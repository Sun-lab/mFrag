#!/usr/bin/env python3
"""CLI: annotated fragments + raw length matrices → mFrag-Weight row-zscored wide CSVs."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
os.environ.setdefault("MFRAG_ROOT", str(CODE_DIR.parent))

from mfrag.preprocessing.pipeline import build_mfrag_weight_inputs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fragments-csv",
        type=Path,
        action="append",
        required=True,
        help="Annotated fragment CSV (repeat for multiple samples)",
    )
    ap.add_argument(
        "--length-raw-dir",
        type=Path,
        required=True,
        help="Directory with fragment_length_raw_weights/*.csv from build_mfrag_net_inputs.py",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--reference-training-csv",
        type=Path,
        action="append",
        default=None,
        help="Training fragment CSVs defining window intersection (optional)",
    )
    args = ap.parse_args()

    outputs = build_mfrag_weight_inputs(
        args.fragments_csv,
        args.length_raw_dir,
        args.out_dir,
        reference_training_csvs=args.reference_training_csv,
    )
    print(f"  sl_ratio_row_zscore:      {outputs.sl_ratio_row_zscore}")
    print(f"  mds_shannon_row_zscore:   {outputs.mds_shannon_row_zscore}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
