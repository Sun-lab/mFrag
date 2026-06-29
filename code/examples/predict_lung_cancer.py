#!/usr/bin/env python3
"""Example: lung cancer probability with mFrag models."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MFRAG_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT.parent / "Mathios" / "scripts"))

from mfrag import MFragNetEnd4Motif, MFragNetFragLength, MFragWeightMDS, MFragWeightSL

SAMPLE = "CGPLLU1P_lung_cancer"
SL_WIDE = ROOT.parent / "Mathios/Mathios_slratio_method/slratio_testing_row_zscore.csv"
MDS_WIDE = ROOT.parent / "Mathios/Mathios_slratio_method/end4motif_shannon_testing_row_zscore.csv"


def main() -> int:
    print(f"Sample: {SAMPLE}\n")

    frag = MFragNetFragLength.from_pretrained()
    motif = MFragNetEnd4Motif.from_pretrained()
    print(f"mFrag-Net (Frag length):   {frag.predict_proba(SAMPLE):.4f}")
    print(f"mFrag-Net (End 4-mer):     {motif.predict_proba(SAMPLE):.4f}")

    try:
        sl = MFragWeightSL.from_pretrained()
        mds = MFragWeightMDS.from_pretrained()
        probs_sl, names = sl.predict_proba_from_wide_csv(SL_WIDE)
        probs_mds, _ = mds.predict_proba_from_wide_csv(MDS_WIDE)
        idx = names.index(SAMPLE)
        print(f"mFrag-Weight (S/L):        {probs_sl[idx]:.4f}")
        print(f"mFrag-Weight (MDS):        {probs_mds[idx]:.4f}")
    except Exception as exc:
        print(f"mFrag-Weight models require scikit-learn>=1.7 (training environment): {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
