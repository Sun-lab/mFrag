#!/usr/bin/env python3
"""
Figure 4 — bootstrap ROC / PR curves.

Frag length: 3 models (DELFI S/L, mFrag-DELFI, mFrag-Net).
End 4-mer motif: 4 models.

Per panel: 4 ROC + 4 PR curves (overall + Stage I / II / III+IV).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

CFCRAFT = Path("/fh/working/sun_w/saishi_cui/cfCraft")
OUT_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure4"
SCRIPT_DIR = Path(__file__).resolve().parent
MATHIOS_SCRIPTS = CFCRAFT / "Mathios" / "scripts"
sys.path.insert(0, str(MATHIOS_SCRIPTS))
from figure4_plot_titles import bootstrap_plot_title  # noqa: E402
from roc_plot_utils import pad_median_iqr_for_origin_vertical, roc_median_iqr  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "fig4_stage",
    SCRIPT_DIR / "plot_figure4_stage_bootstrap_boxplots.py",
)
_stage = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _stage
assert _spec.loader is not None
_spec.loader.exec_module(_stage)

PANELS = _stage.PANELS
STAGE_GROUPS = _stage.STAGE_GROUPS
_load_lasso = _stage._load_lasso
_load_cnn = _stage._load_cnn
_build_stage_pools = _stage._build_stage_pools
_load_panel_models = _stage._load_panel_models
_boot_seed = _stage._boot_seed

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
FONT_LEGEND = 11


def _bootstrap_curves(
    y: np.ndarray,
    prob: np.ndarray,
    pool_idx: np.ndarray | None,
    n_boot: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if pool_idx is None:
        pool = np.arange(len(y), dtype=np.int64)
    else:
        pool = np.asarray(pool_idx, dtype=np.int64)
    n = int(pool.size)
    roc_curves: list[tuple[np.ndarray, np.ndarray]] = []
    pr_curves: list[tuple[np.ndarray, np.ndarray]] = []
    aucs: list[float] = []
    praucs: list[float] = []
    for _ in range(n_boot):
        draw = pool[rng.integers(0, n, size=n)]
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
    panel,
    model_data: dict,
    pool_idx: np.ndarray | None,
    n_boot: int,
    base_seed: int,
    split_slug: str,
) -> tuple[list[tuple], list[tuple], float]:
    roc_bundles = []
    pr_bundles = []
    prevalence = 0.0
    for mkey, _, _, _, seed_tag in panel.models:
        y, prob = model_data[mkey]
        if pool_idx is not None:
            y, prob = y[pool_idx], prob[pool_idx]
        if prevalence == 0.0:
            prevalence = float(y.mean())
        seed_i = _boot_seed(base_seed, panel.slug, split_slug, seed_tag)
        roc_curves, pr_curves, aucs, praucs = _bootstrap_curves(y, prob, None, n_boot, seed_i)
        label = dict(panel.groups)[mkey].replace("\n", " ")
        roc_bundles.append((mkey, label, roc_curves, aucs))
        pr_bundles.append((mkey, label, pr_curves, praucs, prevalence))
    return roc_bundles, pr_bundles, prevalence


def _plot_roc(
    bundles: list[tuple],
    panel,
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
        color = panel.colors[mkey]
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
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        prop={"size": FONT_LEGEND, "weight": "bold"},
    )
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
    panel,
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
        color = panel.colors[mkey]
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
    ax.legend(
        loc="upper right",
        framealpha=0.92,
        prop={"size": FONT_LEGEND, "weight": "bold"},
    )
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_png}")


def _run_panel(panel, *, out_dir: Path, n_boot: int, seed: int, dpi: int, fpr_points: int, recall_points: int) -> None:
    _, _, sample_ids = _load_lasso(panel.models[0][2])
    pools, pool_summary = _build_stage_pools(sample_ids)
    model_data = _load_panel_models(panel)

    for split_key, split_slug, split_label in SPLITS:
        if split_key == "overall":
            pool_idx = None
        else:
            pool_idx = pools[split_key]

        plot_title = bootstrap_plot_title(split_key, split_label, sample_ids, pool_summary)
        roc_bundles, pr_bundles, prevalence = _collect_bundles(
            panel, model_data, pool_idx, n_boot, seed, split_slug
        )
        base = f"figure4_{panel.slug}_{split_slug}"

        _plot_roc(
            roc_bundles,
            panel,
            title=plot_title,
            out_png=out_dir / f"{base}_roc.png",
            out_pdf=out_dir / f"{base}_roc.pdf",
            out_csv=out_dir / f"{base}_roc_summary.csv",
            fpr_points=fpr_points,
            dpi=dpi,
        )
        _plot_pr(
            pr_bundles,
            prevalence,
            panel,
            title=plot_title,
            out_png=out_dir / f"{base}_prroc.png",
            out_pdf=out_dir / f"{base}_prroc.pdf",
            out_csv=out_dir / f"{base}_prroc_summary.csv",
            recall_points=recall_points,
            dpi=dpi,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--panel", choices=("frag_length", "end4motif", "all"), default="all")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fpr-points", type=int, default=2001)
    ap.add_argument("--recall-points", type=int, default=2001)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    _apply_plot_style(dpi=args.dpi)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panels = PANELS if args.panel == "all" else tuple(p for p in PANELS if p.slug == args.panel)
    for panel in panels:
        _run_panel(
            panel,
            out_dir=args.out_dir,
            n_boot=args.n_bootstrap,
            seed=args.seed,
            dpi=args.dpi,
            fpr_points=args.fpr_points,
            recall_points=args.recall_points,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
