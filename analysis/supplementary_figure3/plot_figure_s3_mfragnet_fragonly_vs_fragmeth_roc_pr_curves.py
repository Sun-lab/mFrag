#!/usr/bin/env python3
"""
Figure S3 — bootstrap ROC / PR curves: mFrag-Net frag-only vs frag+meth.

Outputs 4 ROC + 4 PR curves (overall, Stage I, II, III+IV).
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
MATHIOS = CFCRAFT / "Mathios"
FD_DIR = MATHIOS / "FDConv"
META_PATH = MATHIOS / "metadata" / "validation_testing_sample_list.tsv"
MATHIOS_SCRIPTS = MATHIOS / "scripts"
FIG4_DIR = CFCRAFT / "mFrag_Figures" / "Figure4"
OUT_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure S3"

if str(MATHIOS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MATHIOS_SCRIPTS))
if str(FIG4_DIR) not in sys.path:
    sys.path.insert(0, str(FIG4_DIR))

from figure4_plot_titles import bootstrap_plot_title  # noqa: E402
from roc_plot_utils import pad_median_iqr_for_origin_vertical, roc_median_iqr  # noqa: E402

FRAGONLY_NPZ = FD_DIR / "fragonly_new" / "runs" / "rep_135.npz"
FRAG_METH3_NPZ = FD_DIR / "frag_meth3ch_new" / "runs" / "rep_031.npz"

MODELS = (
    ("mfrag_net_fragonly", "mFrag-Net (Frag length without DNA Methyl)", FRAGONLY_NPZ, 135),
    ("mfrag_net_fragmeth", "mFrag-Net (Frag length)", FRAG_METH3_NPZ, 31),
)
COLORS = {
    "mfrag_net_fragonly": "#2ca02c",
    "mfrag_net_fragmeth": "#9467bd",
}

SPLITS = (
    ("overall", "overall", "Overall testing"),
    ("stage_I", "stage_I", "Stage I"),
    ("stage_II", "stage_II", "Stage II"),
    ("stage_III_IV", "stage_III_IV", "Stage III + IV"),
)

FONT_BASE = 14
FONT_TITLE = 15
FONT_LABEL = 15
FONT_TICK = 13
FONT_LEGEND = 10


def _stage_group(stage: str) -> str | None:
    s = str(stage).strip().upper()
    if s == "I":
        return "stage_I"
    if s == "II":
        return "stage_II"
    if s in ("III", "IV"):
        return "stage_III_IV"
    return None


def _load_cnn(npz_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    z = np.load(npz_path, allow_pickle=True)
    y = np.asarray(z["labels"], dtype=np.int32)
    p = np.asarray(z["probs"], dtype=np.float64)
    names = [str(f).replace(".csv", "") for f in z["sample_files"]]
    return y, p, names


def _build_stage_pools(sample_ids: list[str]) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    meta = pd.read_csv(META_PATH, sep="\t")
    meta["cg_id"] = meta["sample_id"].astype(str)
    cg_ids = [s.replace("_healthy", "").replace("_lung_cancer", "") for s in sample_ids]
    y = np.asarray([0 if s.endswith("_healthy") else 1 for s in sample_ids], dtype=np.int32)
    stage_map = meta.set_index("cg_id")["Stage"].astype(str).to_dict()
    stages = [stage_map.get(c, "") for c in cg_ids]
    stage_grp = [_stage_group(st) for st in stages]
    healthy_idx = np.flatnonzero(y == 0)
    pools: dict[str, np.ndarray] = {}
    rows = []
    for split_key, _, split_label in SPLITS:
        if split_key == "overall":
            continue
        cancer_idx = np.flatnonzero((y == 1) & np.array([g == split_key for g in stage_grp]))
        pool = np.sort(np.concatenate([healthy_idx, cancer_idx]))
        pools[split_key] = pool
        rows.append(
            {
                "stage_group": split_key,
                "stage_label": split_label,
                "n_healthy": int(healthy_idx.size),
                "n_cancer_in_stage": int(cancer_idx.size),
                "n_pool": int(pool.size),
            }
        )
    return pools, pd.DataFrame(rows)


def _boot_seed(base_seed: int, split_slug: str, model_tag: int) -> int:
    return base_seed + hash((split_slug, model_tag)) % 10000


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


def _collect_bundles(
    model_data: dict[str, tuple[np.ndarray, np.ndarray]],
    pool_idx: np.ndarray | None,
    n_boot: int,
    base_seed: int,
    split_slug: str,
) -> tuple[list[tuple], list[tuple], float]:
    roc_bundles = []
    pr_bundles = []
    prevalence = 0.0
    for mkey, label, _, seed_tag in MODELS:
        y, prob = model_data[mkey]
        if pool_idx is not None:
            y, prob = y[pool_idx], prob[pool_idx]
        if prevalence == 0.0:
            prevalence = float(y.mean())
        seed_i = _boot_seed(base_seed, split_slug, seed_tag)
        roc_curves, pr_curves, aucs, praucs = _bootstrap_curves(y, prob, n_boot, seed_i)
        roc_bundles.append((mkey, label, roc_curves, aucs))
        pr_bundles.append((mkey, label, pr_curves, praucs, prevalence))
    return roc_bundles, pr_bundles, prevalence


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fpr-points", type=int, default=2001)
    ap.add_argument("--recall-points", type=int, default=2001)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    _apply_plot_style(dpi=args.dpi)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    sample_ids: list[str] | None = None
    for mkey, _, npz_path, _ in MODELS:
        y, prob, names = _load_cnn(npz_path)
        if sample_ids is None:
            sample_ids = names
        elif names != sample_ids:
            raise SystemExit(f"sample_id order mismatch for {mkey}")
        model_data[mkey] = (y, prob)

    pools, pool_summary = _build_stage_pools(sample_ids)
    pool_summary.to_csv(args.out_dir / "figure_s3_mfragnet_roc_pr_stage_pool_sizes.csv", index=False)

    for split_key, split_slug, split_label in SPLITS:
        pool_idx = None if split_key == "overall" else pools[split_key]
        plot_title = bootstrap_plot_title(split_key, split_label, sample_ids, pool_summary)
        roc_bundles, pr_bundles, prevalence = _collect_bundles(
            model_data, pool_idx, args.n_bootstrap, args.seed, split_slug
        )
        base = f"figure_s3_mfragnet_fragonly_vs_fragmeth_{split_slug}"
        _plot_roc(
            roc_bundles,
            title=plot_title,
            out_png=args.out_dir / f"{base}_roc.png",
            out_pdf=args.out_dir / f"{base}_roc.pdf",
            out_csv=args.out_dir / f"{base}_roc_summary.csv",
            fpr_points=args.fpr_points,
            dpi=args.dpi,
        )
        _plot_pr(
            pr_bundles,
            prevalence,
            title=plot_title,
            out_png=args.out_dir / f"{base}_prroc.png",
            out_pdf=args.out_dir / f"{base}_prroc.pdf",
            out_csv=args.out_dir / f"{base}_prroc_summary.csv",
            recall_points=args.recall_points,
            dpi=args.dpi,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
