#!/usr/bin/env python3
"""Export processed matrices, model predictions, and bootstrap summaries for GitHub release."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from mfrag.mfrag_net import MFragNetEnd4Motif, MFragNetFragLength
from mfrag.paths import (
    DATA_DIR,
    LUNG_TEST_FRAG,
    LUNG_TEST_HREF,
    LUNG_TEST_LREF,
    LUNG_TEST_MOTIF,
)

MATHIOS = ROOT.parent / "Mathios"
SL_MODEL = MATHIOS / "Mathios_slratio_method/pca66_l1_coupling_abs_dbeta_over_mean"
MDS_MODEL = MATHIOS / "Mathios_slratio_method/shannon_pca54_l1_coupling_one_minus_lambda"
SL_WIDE_TR = MATHIOS / "Mathios_slratio_method/slratio_training_row_zscore.csv"
SL_WIDE_TE = MATHIOS / "Mathios_slratio_method/slratio_testing_row_zscore.csv"
MDS_WIDE_TR = MATHIOS / "Mathios_slratio_method/end4motif_shannon_training_row_zscore.csv"
MDS_WIDE_TE = MATHIOS / "Mathios_slratio_method/end4motif_shannon_testing_row_zscore.csv"
FINAL24 = MATHIOS / "Final_data_24"

BOOTSTRAP_SOURCES = {
    "mfrag_weight_sl_ratio": MATHIOS / "Mathios_slratio_method/pca66_bootstrap_compare",
    "mfrag_weight_mds": MATHIOS / "Mathios_slratio_method/shannon_pca54_bootstrap_compare",
    "mfrag_net_frag_length": MATHIOS / "FDConv/frag_meth3ch_new_rep031_bootstrap_compare",
    "mfrag_net_end4motif": MATHIOS / "FDConv/motif_meth3ch_new_rep006_bootstrap_compare",
    "four_model_frag_length": MATHIOS / "FDConv/boxplot_lasso_cnn_fragment_length_bootstrap_compare",
    "four_model_end4motif": MATHIOS / "FDConv/boxplot_lasso_cnn_end4motif_bootstrap_compare",
}

MATRIX_MAP = {
    "training/fragment_length_zscored": FINAL24 / "training/training_fragment_length_zscored",
    "training/end4motif_zscored": FINAL24 / "training/training_end4motif_zscored",
    "training/healthy_ref_methylation_zscored": FINAL24 / "training/training_DNAMethylation_HealthyRef_zscored",
    "training/lung_ref_methylation_zscored": FINAL24 / "training/training_DNAMethylation_LungcancerRef_zscored",
    "testing/fragment_length_zscored": FINAL24 / "testing/testing_fragment_length_zscored",
    "testing/end4motif_zscored": FINAL24 / "testing/testing_end4motif_zscored",
    "testing/healthy_ref_methylation_zscored": FINAL24 / "testing/testing_DNAMethylation_HealthyRef_zscored",
    "testing/lung_ref_methylation_zscored": FINAL24 / "testing/testing_DNAMethylation_LungcancerRef_zscored",
}


def _label_from_stem(stem: str) -> int:
    return 1 if "lung_cancer" in stem else 0


def _copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  copied {dst.relative_to(ROOT)}")


def _copy_tree_csv_only(src: Path, dst: Path) -> int:
    if not src.is_dir():
        raise FileNotFoundError(src)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.glob("*.csv")):
        if src.resolve() == (dst / p.name).resolve():
            continue
        shutil.copy2(p, dst / p.name)
        n += 1
    print(f"  copied {n} CSVs -> {dst.relative_to(ROOT)}")
    return n


def export_mfrag_weight_tables(out_root: Path) -> None:
    print("mFrag-Weight wide tables")
    dest = out_root / "mfrag_weight"
    for src, name in (
        (SL_WIDE_TR, "slratio_training_row_zscore.csv"),
        (SL_WIDE_TE, "slratio_testing_row_zscore.csv"),
        (MDS_WIDE_TR, "end4motif_shannon_training_row_zscore.csv"),
        (MDS_WIDE_TE, "end4motif_shannon_testing_row_zscore.csv"),
    ):
        _copy_file(src, dest / name)


def export_predictions(out_root: Path, *, run_cnn: bool = True) -> None:
    print("Model predictions")
    dest = out_root / "predictions"
    dest.mkdir(parents=True, exist_ok=True)
    _copy_file(SL_MODEL / "test_predictions.csv", dest / "mfrag_weight_sl_ratio_testing.csv")
    _copy_file(SL_MODEL / "train_predictions.csv", dest / "mfrag_weight_sl_ratio_training.csv")
    _copy_file(MDS_MODEL / "test_predictions.csv", dest / "mfrag_weight_mds_testing.csv")
    _copy_file(MDS_MODEL / "train_predictions.csv", dest / "mfrag_weight_mds_training.csv")

    if not run_cnn:
        return

    for model_cls, main_dir, h_dir, l_dir, out_name in (
        (MFragNetFragLength, LUNG_TEST_FRAG, LUNG_TEST_HREF, LUNG_TEST_LREF, "mfrag_net_frag_length_testing.csv"),
        (MFragNetEnd4Motif, LUNG_TEST_MOTIF, LUNG_TEST_HREF, LUNG_TEST_LREF, "mfrag_net_end4motif_testing.csv"),
    ):
        out_path = dest / out_name
        if out_path.is_file():
            print(f"  skip existing {out_path.relative_to(ROOT)}")
            continue
        print(f"  running {model_cls.__name__} on testing cohort...")
        model = model_cls.from_pretrained(main_dir=main_dir, healthy_ref_dir=h_dir, lung_ref_dir=l_dir)
        rows = []
        for p in sorted(main_dir.glob("*.csv")):
            stem = p.stem
            rows.append(
                {
                    "sample_id": stem,
                    "y_true": _label_from_stem(stem),
                    "predicted_proba_class1": model.predict_proba(stem),
                }
            )
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  wrote {out_path.relative_to(ROOT)} ({len(rows)} samples)")


def export_bootstrap(out_root: Path) -> None:
    print("Bootstrap summaries")
    dest = out_root / "bootstrap"
    for name, src in BOOTSTRAP_SOURCES.items():
        d = dest / name
        d.mkdir(parents=True, exist_ok=True)
        for pattern in ("bootstrap_summary.csv", "bootstrap_draws.csv", "boxplot_summary.csv"):
            for p in src.glob(pattern):
                shutil.copy2(p, d / p.name)
        for p in sorted(src.glob("*_summary.csv")):
            if p.name in {"bootstrap_summary.csv", "boxplot_summary.csv"}:
                continue
            shutil.copy2(p, d / p.name)
        print(f"  bundled {name} ({len(list(d.glob('*.csv')))} files)")


def export_matrices(out_root: Path, *, split: str | None = None) -> None:
    print("Fragmentomic z-scored matrices")
    lung = out_root / "lung_cancer"
    for rel, src in MATRIX_MAP.items():
        if split and not rel.startswith(split):
            continue
        _copy_tree_csv_only(src, lung / rel)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--skip-matrices", action="store_true")
    ap.add_argument("--matrices-split", choices=["training", "testing"], default=None)
    ap.add_argument("--skip-cnn-predictions", action="store_true")
    args = ap.parse_args()

    export_mfrag_weight_tables(args.out_dir)
    export_predictions(args.out_dir, run_cnn=not args.skip_cnn_predictions)
    export_bootstrap(args.out_dir)
    if not args.skip_matrices:
        export_matrices(args.out_dir, split=args.matrices_split)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
