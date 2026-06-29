#!/usr/bin/env python3
"""
Figure S6 — ROC and PR curves for three testing models.

1. mFrag-DELFI (Frag length)  — PCA66 Lasso |Δβ|/mean
2. mFrag-Net (Frag length)    — frag_meth3ch_new rep031
3. Fragle (ctDNA %)           — Fragle ctDNA fraction
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

CFCRAFT = Path("/fh/working/sun_w/saishi_cui/cfCraft")
MATHIOS = CFCRAFT / "Mathios"
SL_DIR = MATHIOS / "Mathios_slratio_method"
FD_DIR = MATHIOS / "FDConv"
FRAGLE_DIR = CFCRAFT / "Fragle"
MATHIOS_SCRIPTS = MATHIOS / "scripts"
FIG4_DIR = CFCRAFT / "mFrag_Figures" / "Figure4"
OUT_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure S6"

if str(MATHIOS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MATHIOS_SCRIPTS))
if str(FIG4_DIR) not in sys.path:
    sys.path.insert(0, str(FIG4_DIR))
from figure4_plot_titles import bootstrap_plot_title  # noqa: E402
from roc_plot_utils import pad_median_iqr_for_origin_vertical, roc_median_iqr  # noqa: E402

MFRAG_DELFI_PRED = SL_DIR / "pca66_l1_coupling_abs_dbeta_over_mean" / "test_predictions.csv"
MFRAGNET_NPZ = FD_DIR / "frag_meth3ch_new" / "runs" / "rep_031.npz"
FRAGLE_SUMMARY = FRAGLE_DIR / "fragle_ctdna_predictions_summary.csv"
FRAGLE_MANIFEST = FRAGLE_DIR / "bam_manifest.tsv"

GROUPS = (
    ("mfrag_delfi", "mFrag-DELFI (Frag length)"),
    ("mfrag_net", "mFrag-Net (Frag length)"),
    ("fragle", "Fragle (ctDNA %)"),
)
GROUP_LABELS = {k: v for k, v in GROUPS}
COLORS = {
    "mfrag_delfi": "#ff7f0e",
    "mfrag_net": "#9467bd",
    "fragle": "#2ca02c",
}

FONT_TITLE = 17
FONT_LABEL = 15
FONT_TICK = 13
FONT_LEGEND = 10


def _load_mfrag_delfi() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(MFRAG_DELFI_PRED)
    return df["y_true"].to_numpy(dtype=np.int32), df["predicted_proba_class1"].to_numpy(dtype=np.float64)


def _load_mfrag_net() -> tuple[np.ndarray, np.ndarray]:
    z = np.load(MFRAGNET_NPZ, allow_pickle=True)
    return np.asarray(z["labels"], dtype=np.int32), np.asarray(z["probs"], dtype=np.float64)


def _load_fragle() -> tuple[np.ndarray, np.ndarray]:
    summary = pd.read_csv(FRAGLE_SUMMARY)
    manifest = pd.read_csv(FRAGLE_MANIFEST, sep="\t")
    df = summary.merge(manifest[["sample_id", "split"]], on="sample_id", how="inner")
    df = df[(df["split"] == "testing") & (df["group"].isin(["healthy", "cancer"]))].copy()
    y = (df["group"] == "cancer").astype(np.int32).to_numpy()
    score = df["ctdna_fraction"].to_numpy(dtype=np.float64)
    return y, score


def _load_testing_sample_ids() -> list[str]:
    meta = pd.read_csv(FRAGLE_MANIFEST, sep="\t")
    return meta.loc[meta["split"] == "testing", "sample_id"].astype(str).tolist()


def _bootstrap_curves(
    y: np.ndarray,
    score: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[list, list, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(y)
    roc_curves: list[tuple[np.ndarray, np.ndarray]] = []
    pr_curves: list[tuple[np.ndarray, np.ndarray]] = []
    aucs: list[float] = []
    praucs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        s_b = score[idx]
        if len(np.unique(y_b)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_b, s_b)
        prec, rec, _ = precision_recall_curve(y_b, s_b)
        roc_curves.append((fpr, tpr))
        pr_curves.append((prec, rec))
        aucs.append(float(roc_auc_score(y_b, s_b)))
        praucs.append(float(average_precision_score(y_b, s_b)))
    return roc_curves, pr_curves, np.asarray(aucs), np.asarray(praucs)


def _pr_median_iqr(
    curves: list[tuple[np.ndarray, np.ndarray]],
    recall_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mats = []
    for prec, rec in curves:
        order = np.argsort(rec)
        mats.append(np.interp(recall_grid, rec[order], prec[order], left=1.0, right=0.0))
    arr = np.stack(mats, axis=0)
    return (
        np.median(arr, axis=0),
        np.percentile(arr, 25.0, axis=0),
        np.percentile(arr, 75.0, axis=0),
    )


def _plot_roc(
    bundles: list[tuple],
    *,
    title: str,
    out_png: Path,
    out_pdf: Path,
    out_csv: Path,
    fpr_points: int,
    dpi: int,
) -> None:
    fpr_grid = np.linspace(0.0, 1.0, int(fpr_points), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.35, zorder=0)
    rows = []
    for mkey, label, roc_curves, aucs in bundles:
        med, lo, hi = roc_median_iqr(roc_curves, fpr_grid)
        fpr_plot, med, lo, hi = pad_median_iqr_for_origin_vertical(fpr_grid, med, lo, hi)
        auc_med = float(np.median(aucs))
        auc_q1 = float(np.percentile(aucs, 25.0))
        auc_q3 = float(np.percentile(aucs, 75.0))
        color = COLORS[mkey]
        ax.fill_between(fpr_plot, lo, hi, color=color, alpha=0.18, linewidth=0.0, zorder=1)
        ax.plot(
            fpr_plot,
            med,
            color=color,
            lw=2.4,
            label=f"{label} ({auc_med:.3f} [{auc_q1:.3f}, {auc_q3:.3f}])",
            zorder=2,
        )
        rows.append(
            {
                "model_key": mkey,
                "label": label,
                "n_bootstrap": int(aucs.size),
                "auc_median": auc_med,
                "auc_q25": auc_q1,
                "auc_q75": auc_q3,
            }
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False positive rate", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("True positive rate", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, prop={"size": FONT_LEGEND, "weight": "bold"})
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


def _plot_pr(
    bundles: list[tuple],
    prevalence: float,
    *,
    title: str,
    out_png: Path,
    out_pdf: Path,
    out_csv: Path,
    recall_points: int,
    dpi: int,
) -> None:
    recall_grid = np.linspace(0.0, 1.0, int(recall_points), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    rows = []
    for mkey, label, pr_curves, praucs, _prev in bundles:
        med, lo, hi = _pr_median_iqr(pr_curves, recall_grid)
        pr_med = float(np.median(praucs))
        pr_q1 = float(np.percentile(praucs, 25.0))
        pr_q3 = float(np.percentile(praucs, 75.0))
        color = COLORS[mkey]
        ax.fill_between(recall_grid, lo, hi, color=color, alpha=0.18, linewidth=0.0, zorder=1)
        ax.plot(
            recall_grid,
            med,
            color=color,
            lw=2.4,
            label=f"{label} ({pr_med:.3f} [{pr_q1:.3f}, {pr_q3:.3f}])",
            zorder=2,
        )
        rows.append(
            {
                "model_key": mkey,
                "label": label,
                "n_bootstrap": int(praucs.size),
                "prauc_median": pr_med,
                "prauc_q25": pr_q1,
                "prauc_q75": pr_q3,
            }
        )
    ax.axhline(prevalence, color="k", ls="--", lw=1, alpha=0.4, label=f"prevalence = {prevalence:.3f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.legend(loc="upper right", framealpha=0.92, prop={"size": FONT_LEGEND, "weight": "bold"})
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fpr-points", type=int, default=2001)
    ap.add_argument("--recall-points", type=int, default=2001)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_title = bootstrap_plot_title("overall", "All stages", _load_testing_sample_ids())

    loaders = (
        ("mfrag_delfi", GROUP_LABELS["mfrag_delfi"], _load_mfrag_delfi, 66),
        ("mfrag_net", GROUP_LABELS["mfrag_net"], _load_mfrag_net, 31),
        ("fragle", GROUP_LABELS["fragle"], _load_fragle, 0),
    )

    roc_bundles: list[tuple] = []
    pr_bundles: list[tuple] = []
    prevalence = 0.0
    for mkey, label, loader, seed_off in loaders:
        y, score = loader()
        if prevalence == 0.0:
            prevalence = float(y.mean())
        roc_c, pr_c, aucs, praucs = _bootstrap_curves(
            y, score, args.n_bootstrap, args.seed + seed_off
        )
        roc_bundles.append((mkey, label, roc_c, aucs))
        pr_bundles.append((mkey, label, pr_c, praucs, prevalence))

    _plot_roc(
        roc_bundles,
        title=plot_title,
        out_png=args.out_dir / "figure_s6_mfragdelfi_mfragnet_fragle_roc.png",
        out_pdf=args.out_dir / "figure_s6_mfragdelfi_mfragnet_fragle_roc.pdf",
        out_csv=args.out_dir / "figure_s6_mfragdelfi_mfragnet_fragle_roc_summary.csv",
        fpr_points=args.fpr_points,
        dpi=args.dpi,
    )
    _plot_pr(
        pr_bundles,
        prevalence,
        title=plot_title,
        out_png=args.out_dir / "figure_s6_mfragdelfi_mfragnet_fragle_prroc.png",
        out_pdf=args.out_dir / "figure_s6_mfragdelfi_mfragnet_fragle_prroc.pdf",
        out_csv=args.out_dir / "figure_s6_mfragdelfi_mfragnet_fragle_prroc_summary.csv",
        recall_points=args.recall_points,
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
