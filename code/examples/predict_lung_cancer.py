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
    build_mfrag_net_inputs,
    build_mfrag_weight_inputs,
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

    # --- Preprocessing (when starting from an annotated fragment CSV) ---
    # frag_csv = ROOT / "work" / f"{SAMPLE}.csv"
    # net = build_mfrag_net_inputs(frag_csv, ROOT / "work" / "mfrag_net", write_raw=True)
    # weight = build_mfrag_weight_inputs(
    #     [frag_csv],
    #     ROOT / "work" / "mfrag_net" / "fragment_length_raw_weights",
    #     ROOT / "work" / "mfrag_weight",
    # )

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
    print(f"mFrag-Net (fragment length): {frag_net.predict_proba(SAMPLE):.4f}")
    print(f"mFrag-Net (end 4-mer motif): {motif_net.predict_proba(SAMPLE):.4f}")

    # mFrag-Weight: point to row-zscored wide CSVs produced by build_mfrag_weight_inputs
    sl_wide = ROOT / "work" / "mfrag_weight" / "slratio_row_zscore.csv"
    mds_wide = ROOT / "work" / "mfrag_weight" / "end4motif_shannon_row_zscore.csv"
    if sl_wide.is_file() and mds_wide.is_file():
        sl = MFragWeightSL.from_pretrained()
        mds = MFragWeightMDS.from_pretrained()
        probs_sl, names = sl.predict_proba_from_wide_csv(sl_wide)
        probs_mds, _ = mds.predict_proba_from_wide_csv(mds_wide)
        idx = names.index(SAMPLE)
        print(f"mFrag-Weight (S/L ratio):    {probs_sl[idx]:.4f}")
        print(f"mFrag-Weight (MDS Shannon):  {probs_mds[idx]:.4f}")
    else:
        print(
            "mFrag-Weight: run build_mfrag_weight_inputs() first, or set sl_wide/mds_wide paths."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
