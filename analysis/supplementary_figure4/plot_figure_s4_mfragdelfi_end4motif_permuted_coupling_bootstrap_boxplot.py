#!/usr/bin/env python3
"""
Figure S4 — mFrag-DELFI (End 4-mer motif) = Shannon PCA(54)+Lasso (1−λΔβ̃)
vs three permuted-coupling null draws.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

CFCRAFT = Path("/fh/working/sun_w/saishi_cui/cfCraft")
MATHIOS = CFCRAFT / "Mathios"
SL_DIR = MATHIOS / "Mathios_slratio_method"
SHANNON_BOOT = SL_DIR / "shannon_pca54_bootstrap_compare" / "bootstrap_draws.csv"
PERM_BOOT = SL_DIR / "shannon_pca54_l1_one_minus_lambda_permuted_bootstrap_compare" / "bootstrap_draws.csv"
OUT_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure S4"
SCRIPT_DIR = MATHIOS / "scripts"
FRAG_PLOT = Path(__file__).with_name(
    "plot_figure_s4_mfragdelfi_lasso_permuted_coupling_bootstrap_boxplot.py"
)

GROUPS = (
    ("mfrag_delfi_end4motif", "mFrag-Weight (MDS)"),
    ("permute_1", "mFrag-Weight (MDS random permutated weights 1)"),
    ("permute_2", "mFrag-Weight (MDS random permutated weights 2)"),
    ("permute_3", "mFrag-Weight (MDS random permutated weights 3)"),
)
GROUP_LABELS = {k: v for k, v in GROUPS}
SOURCE_VARIANT = {
    "mfrag_delfi_end4motif": "one_minus_lambda",
    "permute_1": "permute_1",
    "permute_2": "permute_2",
    "permute_3": "permute_3",
}
COLORS = {
    "mfrag_delfi_end4motif": "#ff7f0e",
    "permute_1": "#2ca02c",
    "permute_2": "#9467bd",
    "permute_3": "#1f77b4",
}
REF_KEY = "mfrag_delfi_end4motif"
PERMUTE_KEYS = ("permute_1", "permute_2", "permute_3")


def _load_frag_plot_module():
    spec = importlib.util.spec_from_file_location("s4_frag_plot", FRAG_PLOT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["s4_frag_plot"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_series() -> dict[str, dict[str, np.ndarray]]:
    shannon = pd.read_csv(SHANNON_BOOT)
    ref = shannon[(shannon["family"] == "Lasso") & (shannon["variant"] == "one_minus_lambda")]
    if ref.empty:
        raise SystemExit(f"Missing Lasso one_minus_lambda in {SHANNON_BOOT}")

    perm = pd.read_csv(PERM_BOOT)
    out: dict[str, dict[str, np.ndarray]] = {
        "mfrag_delfi_end4motif": {
            "auc": ref["test_auc"].to_numpy(dtype=np.float64),
            "prauc": ref["test_prauc"].to_numpy(dtype=np.float64),
        }
    }
    for gkey in PERMUTE_KEYS:
        variant = SOURCE_VARIANT[gkey]
        sub = perm[(perm["family"] == "Lasso") & (perm["variant"] == variant)]
        if sub.empty:
            raise SystemExit(f"Missing variant={variant} in {PERM_BOOT}")
        out[gkey] = {
            "auc": sub["test_auc"].to_numpy(dtype=np.float64),
            "prauc": sub["test_prauc"].to_numpy(dtype=np.float64),
        }
    return out


def _mwu_ref_vs_permutes(
    data: dict[str, dict[str, np.ndarray]], *, sig_star, group_labels: dict[str, str]
) -> pd.DataFrame:
    rows = []
    for metric in ("auc", "prauc"):
        ref = data[REF_KEY][metric]
        for pkey in PERMUTE_KEYS:
            comp = data[pkey][metric]
            res = mannwhitneyu(ref, comp, alternative="greater", method="auto")
            rows.append(
                {
                    "metric": metric,
                    "reference_key": REF_KEY,
                    "reference_label": group_labels[REF_KEY],
                    "comparator_key": pkey,
                    "comparator_label": group_labels[pkey],
                    "u_stat": float(res.statistic),
                    "p_value": float(res.pvalue),
                    "median_reference": float(np.median(ref)),
                    "median_comparator": float(np.median(comp)),
                    "alternative": "mFrag-Weight (MDS) > permuted draw",
                    "sig": sig_star(float(res.pvalue)),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--train-csv", type=Path, default=SL_DIR / "end4motif_shannon_training_row_zscore.csv")
    ap.add_argument("--test-csv", type=Path, default=SL_DIR / "end4motif_shannon_testing_row_zscore.csv")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-retrain", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    if not args.skip_train:
        boot_spec = importlib.util.spec_from_file_location(
            "shannon_perm_boot",
            SCRIPT_DIR / "shannon_pca54_l1_one_minus_lambda_permuted_coupling_bootstrap.py",
        )
        boot_mod = importlib.util.module_from_spec(boot_spec)
        assert boot_spec.loader is not None
        sys.modules["shannon_perm_boot"] = boot_mod
        boot_spec.loader.exec_module(boot_mod)
        boot_mod._train_permuted_models(args.train_csv, args.test_csv, force=args.force_retrain)
        argv = [
            "shannon_pca54_l1_one_minus_lambda_permuted_coupling_bootstrap.py",
            "--train-csv",
            str(args.train_csv),
            "--test-csv",
            str(args.test_csv),
            "--n-bootstrap",
            str(args.n_bootstrap),
            "--seed",
            str(args.seed),
            "--skip-train",
        ]
        old_argv = sys.argv
        sys.argv = argv
        try:
            boot_mod.main()
        finally:
            sys.argv = old_argv

    frag = _load_frag_plot_module()
    data = _load_series()
    mwu = _mwu_ref_vs_permutes(data, sig_star=frag._sig_star, group_labels=GROUP_LABELS)

    summary_rows = []
    long_rows = []
    for gkey, glabel in GROUPS:
        for mkey in ("auc", "prauc"):
            arr = data[gkey][mkey]
            summary_rows.append(
                {
                    "group_key": gkey,
                    "group_label": glabel,
                    "source_variant": SOURCE_VARIANT[gkey],
                    "metric": mkey,
                    "n": len(arr),
                    "median": float(np.median(arr)),
                    "q25": float(np.percentile(arr, 25)),
                    "q75": float(np.percentile(arr, 75)),
                }
            )
            for i, v in enumerate(arr):
                long_rows.append(
                    {
                        "group_key": gkey,
                        "group_label": glabel,
                        "source_variant": SOURCE_VARIANT[gkey],
                        "metric": mkey,
                        "bootstrap_idx": i,
                        "value": float(v),
                    }
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(
        args.out_dir / "figure_s4_mfragdelfi_end4motif_permuted_coupling_bootstrap_summary.csv",
        index=False,
    )
    pd.DataFrame(long_rows).to_csv(
        args.out_dir / "figure_s4_mfragdelfi_end4motif_permuted_coupling_bootstrap_draws.csv",
        index=False,
    )
    mwu.to_csv(
        args.out_dir / "figure_s4_mfragdelfi_end4motif_permuted_coupling_bootstrap_mwu.csv",
        index=False,
    )
    print(mwu.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    frag.GROUPS = GROUPS
    frag.GROUP_LABELS = GROUP_LABELS
    frag.COLORS = COLORS
    frag.REF_KEY = REF_KEY
    frag.PERMUTE_KEYS = PERMUTE_KEYS
    frag.BRACKET_Y = {"auc": (1.36, 1.22, 1.06), "prauc": (1.09, 0.95, 0.79)}
    frag.BRACKET_LINE_GAP_MUL = {"auc": (1.0, 0.55, 0.62), "prauc": (1.0, 0.55, 0.62)}
    frag.MIN_STAR_P_GAP = {"auc": (0.0, 0.0, 0.048), "prauc": (0.0, 0.0, 0.048)}

    plot_title = "All stages (testing healthy = 385, testing cancer = 46)"
    frag._plot(
        data=data,
        out_png=args.out_dir / "figure_s4_mfragdelfi_end4motif_permuted_coupling_bootstrap_boxplot.png",
        out_pdf=args.out_dir / "figure_s4_mfragdelfi_end4motif_permuted_coupling_bootstrap_boxplot.pdf",
        ylim=(0.0, 1.7),
        dpi=args.dpi,
        mwu=mwu,
        plot_title=plot_title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
