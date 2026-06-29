#!/usr/bin/env python3
"""CLI: annotated fragment CSV → four z-scored 589×24 matrices for mFrag-Net."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
os.environ.setdefault("MFRAG_ROOT", str(CODE_DIR.parent))

from mfrag.preprocessing.pipeline import build_mfrag_net_inputs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fragments-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--no-raw",
        action="store_true",
        help="Skip writing raw-weight matrices (mFrag-Weight needs raw length matrices)",
    )
    args = ap.parse_args()

    outputs = build_mfrag_net_inputs(
        args.fragments_csv,
        args.out_dir,
        write_raw=not args.no_raw,
    )
    print(f"sample: {outputs.sample_stem}")
    print(f"  fragment_length_zscored:          {outputs.fragment_length_zscored}")
    print(f"  end4motif_zscored:                {outputs.end4motif_zscored}")
    print(f"  healthy_ref_methylation_zscored:  {outputs.healthy_ref_methylation_zscored}")
    print(f"  lung_ref_methylation_zscored:     {outputs.lung_ref_methylation_zscored}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
