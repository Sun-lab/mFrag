#!/usr/bin/env python3
"""Example: preprocess fragments and predict lung-cancer probability with mFrag."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MFRAG_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from mfrag import (
    MFragNetEnd4Motif,
    MFragNetFragLength,
    MFragWeightMDS,
    MFragWeightSL,
    average_proba,
)
from mfrag.paths import (
    LUNG_TEST_FRAG,
    LUNG_TEST_HREF,
    LUNG_TEST_LREF,
    LUNG_TEST_MOTIF,
)

# Example sample with bundled lung-cancer testing matrices (symlinked in dev checkout)
SAMPLE = "CGPLLU1P_lung_cancer"


def main() -> int:
    print("mFrag lung-cancer inference demo\n")
    print(f"Sample: {SAMPLE}\n")

    # --- Inference using pre-built testing matrices ---
    frag_net = MFragNetFragLength.from_pretrained(
        main_dir=LUNG_TEST_FRAG,
        healthy_ref_dir=LUNG_TEST_HREF,
        lung_ref_dir=LUNG_TEST_LREF,
    )
    motif_net = MFragNetEnd4Motif.from_pretrained(
        main_dir=LUNG_TEST_MOTIF,
        healthy_ref_dir=LUNG_TEST_HREF,
        lung_ref_dir=LUNG_TEST_LREF,
    )
    p_net_frag = float(frag_net.predict_proba(SAMPLE))
    p_net_motif = float(motif_net.predict_proba(SAMPLE))
    print(f"mFrag-Net (fragment length): {p_net_frag:.4f}")
    print(f"mFrag-Net (end 4-mer motif): {p_net_motif:.4f}")

    # mFrag-Weight: point to row-zscored wide CSVs produced by build_mfrag_weight_inputs
    sl_wide = ROOT / "work" / "mfrag_weight" / "slratio_row_zscore.csv"
    mds_wide = ROOT / "work" / "mfrag_weight" / "end4motif_shannon_row_zscore.csv"
    if sl_wide.is_file() and mds_wide.is_file():
        sl = MFragWeightSL.from_pretrained()
        mds = MFragWeightMDS.from_pretrained()
        probs_sl, names = sl.predict_proba_from_wide_csv(sl_wide)
        probs_mds, _ = mds.predict_proba_from_wide_csv(mds_wide)
        idx = names.index(SAMPLE)
        p_w_sl = float(probs_sl[idx])
        p_w_mds = float(probs_mds[idx])
        print(f"mFrag-Weight (S/L ratio):    {p_w_sl:.4f}")
        print(f"mFrag-Weight (MDS Shannon):  {p_w_mds:.4f}")

        # Primary manuscript scores: equal-weight average per modality
        p_mfrag_frag = average_proba(p_w_sl, p_net_frag)
        p_mfrag_motif = average_proba(p_w_mds, p_net_motif)
        print(f"\nmFrag (Frag length) = 0.5*(Weight S/L + Net frag): {p_mfrag_frag:.4f}")
        print(f"mFrag (Motif)       = 0.5*(Weight MDS + Net motif): {p_mfrag_motif:.4f}")
    else:
        print(
            "mFrag-Weight: run build_mfrag_weight_inputs() first, or set sl_wide/mds_wide paths."
        )
        print(
            "\nOnce both probabilities are available, the primary mFrag score is:\n"
            "  from mfrag import average_proba\n"
            "  mfrag_score = average_proba(weight_proba, net_proba)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
