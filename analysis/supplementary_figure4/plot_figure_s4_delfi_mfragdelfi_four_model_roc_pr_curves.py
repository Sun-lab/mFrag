#!/usr/bin/env python3
"""
Figure S4 — paired ROC / PR curves for four-model DELFI / mFrag-DELFI comparison.

Four figures (overall testing, n=431):
  1. DELFI (S/L ratio) vs DELFI (Concatenation) — ROC
  2. DELFI (S/L ratio) vs DELFI (Concatenation) — PR-ROC
  3. mFrag-DELFI (Frag length) vs mFrag-DELFI (Concatenation) — ROC
  4. mFrag-DELFI (Frag length) vs mFrag-DELFI (Concatenation) — PR-ROC
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

CFCRAFT = Path("/fh/working/sun_w/saishi_cui/cfCraft")
SL_DIR = CFCRAFT / "Mathios" / "Mathios_slratio_method"
MATHIOS_SCRIPTS = CFCRAFT / "Mathios" / "scripts"
OUT_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure S4"

if str(MATHIOS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MATHIOS_SCRIPTS))
from roc_plot_utils import pad_median_iqr_for_origin_vertical, roc_median_iqr  # noqa: E402

PAIRS = (
    {
        "slug": "delfi_sl_vs_concat",
        "file_prefix": "figure_s4_delfi_sl_vs_concat",
        "models": (
            (
                "delfi_sl_ratio",
                "DELFI (S/L ratio)",
                SL_DIR / "pca66_l1_baseline" / "test_predictions.csv",
                0,
            ),
            (
                "delfi_concatenation",
                "DELFI (S/L ratio + MDS)",
                SL_DIR / "concat_slratio_shannon_pca89_l1_baseline" / "test_predictions.csv",
                1,
            ),
        ),
    },
    {
        "slug": "mfragdelfi_frag_vs_concat",
        "file_prefix": "figure_s4_mfragdelfi_frag_vs_concat",
        "models": (
            (
                "mfrag_delfi_frag_length",
                "mFrag-Weight (S/L ratio)",
                SL_DIR / "pca66_l1_coupling_abs_dbeta_over_mean" / "test_predictions.csv",
                0,
            ),
            (
                "mfrag_delfi_concatenation",
                "mFrag-Weight (S/L ratio + MDS)",
                SL_DIR / "concat_slratio_shannon_pca89_l1_coupling" / "test_predictions.csv",
                1,
            ),
        ),
    },
)

COLORS = {
    "delfi_sl_ratio": "#1f77b4",
    "delfi_concatenation": "#17becf",
    "mfrag_delfi_frag_length": "#ff7f0e",
    "mfrag_delfi_concatenation": "#2ca02c",
}

FONT_BASE = 14
FONT_TITLE = 15
FONT_LABEL = 15
FONT_TICK = 13
FONT_LEGEND = 10
PLOT_TITLE = "All stages (testing healthy = 385, testing cancer = 46)"


def _load_pred(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    return (
        df["y_true"].to_numpy(dtype=np.int32),
        df["predicted_proba_class1"].to_numpy(dtype=np.float64),
    )


def _boot_seed(base_seed: int, pair_slug: str, model_tag: int) -> int:
    return base_seed + hash((pair_slug, model_tag)) % 10000


def _bootstrap_curves(
    y: np.ndarray,
    prob: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(y.size)
    roc_curves: list[tuple[np.ndarray, np.ndarray]] = []
    pr_curves: list[tuple[np.ndarray, np.ndarray]] = []
    aucs: list[float] = []
    praucs: list[float] = []
    for _ in range(n_boot):
        draw = rng.integers(0, n, size=n)
        y_b, p_b = y[draw], prob[draw]
        if len(np.unique(y_b)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_b, p_b)
        prec, rec, _ = precision_recall_curve(y_b, p_b)
        roc_curves.append((fpr, tpr))
        pr_curves.append((prec, rec))
        aucs.append(float(roc_auc_score(y_b, p_b)))
        praucs.append(float(average_precision_score(y_b, p_b)))
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


def _apply_plot_style(*, dpi: int) -> None:
    mpl.rcParams.update(
        {
            "font.size": FONT_BASE,
            "axes.titlesize": FONT_TITLE,
            "axes.labelsize": FONT_LABEL,
            "legend.fontsize": FONT_LEGEND,
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.linewidth": 1.0,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _plot_roc(
    bundles: list[tuple],
    *,
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
    ax.set_title(PLOT_TITLE, fontsize=FONT_TITLE, fontweight="bold")
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


def _plot_pr(
    bundles: list[tuple],
    prevalence: float,
    *,
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
    ax.set_title(PLOT_TITLE, fontsize=FONT_TITLE, fontweight="bold")
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


def _run_pair(
    pair: dict,
    *,
    out_dir: Path,
    n_boot: int,
    base_seed: int,
    fpr_points: int,
    recall_points: int,
    dpi: int,
) -> None:
    slug = pair["slug"]
    prefix = pair["file_prefix"]
    roc_bundles = []
    pr_bundles = []
    prevalence = 0.0
    y_ref: np.ndarray | None = None

    for mkey, label, pred_csv, tag in pair["models"]:
        y, prob = _load_pred(pred_csv)
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y_ref, y):
            raise SystemExit(f"y_true mismatch for {pred_csv}")
        prevalence = float(y.mean())
        seed_i = _boot_seed(base_seed, slug, tag)
        roc_curves, pr_curves, aucs, praucs = _bootstrap_curves(y, prob, n_boot, seed_i)
        roc_bundles.append((mkey, label, roc_curves, aucs))
        pr_bundles.append((mkey, label, pr_curves, praucs, prevalence))

    _plot_roc(
        roc_bundles,
        out_png=out_dir / f"{prefix}_roc.png",
        out_pdf=out_dir / f"{prefix}_roc.pdf",
        out_csv=out_dir / f"{prefix}_roc_summary.csv",
        fpr_points=fpr_points,
        dpi=dpi,
    )
    _plot_pr(
        pr_bundles,
        prevalence,
        out_png=out_dir / f"{prefix}_prroc.png",
        out_pdf=out_dir / f"{prefix}_prroc.pdf",
        out_csv=out_dir / f"{prefix}_prroc_summary.csv",
        recall_points=recall_points,
        dpi=dpi,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fpr-points", type=int, default=2001)
    ap.add_argument("--recall-points", type=int, default=2001)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--pair", choices=[p["slug"] for p in PAIRS] + ["all"], default="all")
    args = ap.parse_args()

    _apply_plot_style(dpi=args.dpi)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [p for p in PAIRS if args.pair in ("all", p["slug"])]
    for pair in pairs:
        _run_pair(
            pair,
            out_dir=args.out_dir,
            n_boot=args.n_bootstrap,
            base_seed=args.seed,
            fpr_points=args.fpr_points,
            recall_points=args.recall_points,
            dpi=args.dpi,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
