#!/usr/bin/env python3
"""
Figure5: DELFI (End 4-mer motif) vs mFrag-Net (End 4-mer motif) diagnostic plots.

mFrag-Net = motif_meth3ch_new rep_006 (reference DNA Methyl), matching Figure4.
Outputs: dual forest (all testing) + Stage I paired scatter + Stage I precision@top-k.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import average_precision_score, roc_auc_score

CFCRAFT = Path("/fh/working/sun_w/saishi_cui/cfCraft")
MATHIOS = CFCRAFT / "Mathios"
SL_DIR = MATHIOS / "Mathios_slratio_method"
FD_DIR = MATHIOS / "FDConv"
META_PATH = MATHIOS / "metadata" / "validation_testing_sample_list.tsv"
OUT_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure5"

DELFI_CSV = SL_DIR / "shannon_pca54_l1_baseline" / "test_predictions.csv"
MFRAGNET_NPZ = FD_DIR / "motif_meth3ch_new" / "runs" / "rep_006.npz"

LABEL_DELFI = "DELFI (End 4-mer motif)"
LABEL_MFRAGNET = "mFrag-Net (End 4-mer motif)"
COLOR_HEALTHY = "#aec7e8"
COLOR_DELFI = "#1f77b4"
COLOR_MFRAGNET = "#9467bd"
COLOR_STAGE_I = "#d62728"
STAGE_COLORS = {
    "I": "#d62728",
    "II": "#ff7f0e",
    "III": "#2ca02c",
    "IV": "#9467bd",
}
STAGE_LABELS = {
    "I": "Stage I",
    "II": "Stage II",
    "III": "Stage III",
    "IV": "Stage IV",
}
STAGE_KEY = "stage_I"
STAGE_LABEL = "Stage I"
FILE_PREFIX = "figure5_end4motif_delfi_vs_mfragnet"


def _apply_style(*, dpi: int = 300) -> None:
    mpl.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _stage_group(stage: str) -> str | None:
    s = str(stage).strip().upper()
    if s == "I":
        return "stage_I"
    if s == "II":
        return "stage_II"
    if s in ("III", "IV"):
        return "stage_III_IV"
    return None


def _load_cnn(npz_path: Path) -> pd.DataFrame:
    z = np.load(npz_path, allow_pickle=True)
    names = [str(f).replace(".csv", "") for f in z["sample_files"]]
    return pd.DataFrame(
        {
            "sample_id": names,
            "y_true": np.asarray(z["labels"], dtype=np.int32),
            "predicted_proba_class1_mfragnet": np.asarray(z["probs"], dtype=np.float64),
        }
    )


def load_merged() -> pd.DataFrame:
    d = pd.read_csv(DELFI_CSV).rename(columns={"predicted_proba_class1": "predicted_proba_class1_delfi"})
    m = _load_cnn(MFRAGNET_NPZ)
    df = d.merge(m, on=["sample_id", "y_true"])
    df["delta"] = df["predicted_proba_class1_mfragnet"] - df["predicted_proba_class1_delfi"]
    df["cg"] = (
        df["sample_id"]
        .astype(str)
        .str.replace("_healthy", "", regex=False)
        .str.replace("_lung_cancer", "", regex=False)
    )
    meta = pd.read_csv(META_PATH, sep="\t")
    meta["cg_id"] = meta["sample_id"].astype(str)
    df = df.merge(meta[["cg_id", "Stage"]], left_on="cg", right_on="cg_id", how="left")
    df["stage_grp"] = df["Stage"].map(lambda s: _stage_group(s) if pd.notna(s) else None)
    df["short_id"] = df["sample_id"].str.replace("_healthy", "").str.replace("_lung_cancer", "")
    return df


def subset_stage(df: pd.DataFrame, stage_key: str) -> pd.DataFrame:
    healthy = df["y_true"] == 0
    cancer = (df["y_true"] == 1) & (df["stage_grp"] == stage_key)
    return df[healthy | cancer].copy()


def _sample_color(row: pd.Series) -> str:
    if row["y_true"] == 0:
        return COLOR_HEALTHY
    stage = str(row["Stage"]).strip().upper() if pd.notna(row["Stage"]) else ""
    return STAGE_COLORS.get(stage, "#7f7f7f")


def _short_id(sample_id: str) -> str:
    return sample_id.replace("_healthy", "").replace("_lung_cancer", "")


def plot_dual_forest(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    work = df.copy()
    work["point_color"] = work.apply(_sample_color, axis=1)
    healthy = work[work["y_true"] == 0].sort_values("predicted_proba_class1_delfi")
    cancer = work[work["y_true"] == 1].sort_values("predicted_proba_class1_delfi")
    plot_df = pd.concat([healthy, cancer], ignore_index=True)
    n_h, n_c = len(healthy), len(cancer)
    gap = 2.5
    y_h = np.arange(n_h, dtype=float)
    y_c = np.arange(n_c, dtype=float) + n_h + gap
    y_all = np.concatenate([y_h, y_c])
    colors = plot_df["point_color"].to_numpy()
    labels = [_short_id(s) for s in plot_df["sample_id"]]

    fig_h = max(12, 0.16 * len(plot_df) + 2)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.scatter(
        plot_df["predicted_proba_class1_delfi"],
        y_all,
        s=30,
        c=colors,
        alpha=0.9,
        marker="o",
        edgecolors="0.25",
        linewidths=0.4,
        zorder=3,
    )
    ax.scatter(
        plot_df["predicted_proba_class1_mfragnet"],
        y_all,
        s=30,
        c=colors,
        alpha=0.9,
        marker="D",
        edgecolors="0.25",
        linewidths=0.4,
        zorder=4,
    )
    for yi, d, m in zip(
        y_all,
        plot_df["predicted_proba_class1_delfi"],
        plot_df["predicted_proba_class1_mfragnet"],
    ):
        ax.plot([d, m], [yi, yi], color="black", lw=1.8, zorder=2)

    if n_h and n_c:
        ax.axhline(n_h + gap * 0.5 - 0.5, color="black", lw=2.8, ls="--", zorder=1)

    stage_handles = [Patch(facecolor=COLOR_HEALTHY, edgecolor="0.25", label=f"healthy (n={n_h})")]
    for stage_key in ("I", "II", "III", "IV"):
        n_stage = int((cancer["Stage"].astype(str).str.upper() == stage_key).sum())
        if n_stage:
            stage_handles.append(
                Patch(
                    facecolor=STAGE_COLORS[stage_key],
                    edgecolor="0.25",
                    label=f"{STAGE_LABELS[stage_key]} (n={n_stage})",
                )
            )
    model_handles = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="0.45",
            markeredgecolor="0.25", markersize=8, label=LABEL_DELFI,
        ),
        Line2D(
            [0], [0], marker="D", color="w", markerfacecolor="0.45",
            markeredgecolor="0.25", markersize=7, label=LABEL_MFRAGNET,
        ),
    ]

    ax.set_xlabel("Predicted probability (class 1 = lung cancer)", fontsize=13, fontweight="bold")
    ax.set_ylabel("")
    ax.set_yticks(y_all)
    ax.set_yticklabels(labels, fontsize=8.5, fontweight="bold")
    ax.set_xlim(-0.02, 1.05)
    ax.tick_params(axis="x", labelsize=12, width=1.2)
    ax.grid(True, axis="x", alpha=0.25)
    leg1 = ax.legend(
        handles=stage_handles,
        loc="upper left",
        framealpha=0.92,
        prop={"weight": "bold", "size": 10},
        title="Sample group",
        title_fontproperties={"weight": "bold", "size": 10},
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=model_handles,
        loc="lower right",
        framealpha=0.92,
        prop={"weight": "bold", "size": 10},
        title="Model",
        title_fontproperties={"weight": "bold", "size": 10},
    )
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")
    fig.tight_layout(rect=[0.18, 0.02, 0.98, 0.98])
    for ext in ("png", "pdf"):
        p = out_dir / f"{FILE_PREFIX}_dual_forest.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=dpi)
        print(f"Wrote {p}")
    plt.close(fig)


def plot_paired_scatter(df: pd.DataFrame, out_dir: Path, dpi: int, stage_key: str, stage_label: str) -> None:
    sub = subset_stage(df, stage_key)
    h = sub[sub["y_true"] == 0]
    c = sub[sub["y_true"] == 1]

    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.35, zorder=0)
    if len(h):
        ax.scatter(
            h["predicted_proba_class1_delfi"],
            h["predicted_proba_class1_mfragnet"],
            s=28,
            c=COLOR_HEALTHY,
            alpha=0.55,
            edgecolors="none",
            label=f"healthy (n={len(h)})",
            zorder=2,
        )
    if len(c):
        ax.scatter(
            c["predicted_proba_class1_delfi"],
            c["predicted_proba_class1_mfragnet"],
            s=36,
            c=COLOR_STAGE_I,
            alpha=0.75,
            edgecolors="none",
            label=f"{stage_label} lung cancer (n={len(c)})",
            zorder=3,
        )

    pr_d = average_precision_score(sub["y_true"], sub["predicted_proba_class1_delfi"])
    pr_m = average_precision_score(sub["y_true"], sub["predicted_proba_class1_mfragnet"])
    auc_d = roc_auc_score(sub["y_true"], sub["predicted_proba_class1_delfi"])
    auc_m = roc_auc_score(sub["y_true"], sub["predicted_proba_class1_mfragnet"])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(LABEL_DELFI)
    ax.set_ylabel(LABEL_MFRAGNET)
    ax.set_title(
        f"{stage_label}: all healthy + {stage_label} cancer\n"
        f"AUC {auc_d:.3f} → {auc_m:.3f}; PRAUC {pr_d:.3f} → {pr_m:.3f}",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.9, prop={"weight": "bold", "size": 9})
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    fig.tight_layout()

    stem = f"{FILE_PREFIX}_{stage_key}"
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}_paired_scatter.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=dpi)
        print(f"Wrote {p}")
    plt.close(fig)


def _precision_at_top_k(y: np.ndarray, prob: np.ndarray, k: int) -> float:
    if k <= 0:
        return np.nan
    order = np.argsort(-prob)
    return float(y[order[:k]].mean())


def plot_precision_at_k(df: pd.DataFrame, out_dir: Path, dpi: int, stage_key: str, stage_label: str) -> None:
    sub = subset_stage(df, stage_key)
    y = sub["y_true"].to_numpy(dtype=np.int32)
    p_d = sub["predicted_proba_class1_delfi"].to_numpy(dtype=np.float64)
    p_m = sub["predicted_proba_class1_mfragnet"].to_numpy(dtype=np.float64)
    n = len(y)
    ks = np.unique(np.round(np.linspace(1, n, 80)).astype(int))
    prec_d = [_precision_at_top_k(y, p_d, k) for k in ks]
    prec_m = [_precision_at_top_k(y, p_m, k) for k in ks]
    prev = float(y.mean())

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(ks, prec_d, color=COLOR_DELFI, lw=2.2, label=LABEL_DELFI)
    ax.plot(ks, prec_m, color=COLOR_MFRAGNET, lw=2.2, label=LABEL_MFRAGNET)
    ax.axhline(prev, color="k", ls="--", lw=1, alpha=0.4, label=f"prevalence = {prev:.3f}")
    ax.set_xlabel("Top k samples flagged", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision among flagged", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{stage_label}: precision @ top-k (healthy + {stage_label} cancer, n={n})",
        fontweight="bold",
    )
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.9, prop={"weight": "bold", "size": 10})
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    fig.tight_layout()

    stem = f"{FILE_PREFIX}_{stage_key}"
    rows = [
        {"k": int(k), "fraction": k / n, "precision_delfi": fd, "precision_mfragnet": fm}
        for k, fd, fm in zip(ks, prec_d, prec_m)
    ]
    pd.DataFrame(rows).to_csv(out_dir / f"{stem}_precision_at_k.csv", index=False)
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}_precision_at_k.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=dpi)
        print(f"Wrote {p}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--stage-key", type=str, default=STAGE_KEY)
    ap.add_argument("--stage-label", type=str, default=STAGE_LABEL)
    args = ap.parse_args()

    _apply_style(dpi=args.dpi)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_merged()
    df.to_csv(args.out_dir / f"{FILE_PREFIX}_per_sample.csv", index=False)

    plot_dual_forest(df, args.out_dir, args.dpi)

    sub = subset_stage(df, args.stage_key)
    stem = f"{FILE_PREFIX}_{args.stage_key}"
    sub.to_csv(args.out_dir / f"{stem}_per_sample.csv", index=False)

    summary = {
        "stage_key": args.stage_key,
        "stage_label": args.stage_label,
        "n": len(sub),
        "n_healthy": int((sub["y_true"] == 0).sum()),
        "n_cancer": int((sub["y_true"] == 1).sum()),
        "auc_delfi": float(roc_auc_score(sub["y_true"], sub["predicted_proba_class1_delfi"])),
        "auc_mfragnet": float(roc_auc_score(sub["y_true"], sub["predicted_proba_class1_mfragnet"])),
        "prauc_delfi": float(average_precision_score(sub["y_true"], sub["predicted_proba_class1_delfi"])),
        "prauc_mfragnet": float(
            average_precision_score(sub["y_true"], sub["predicted_proba_class1_mfragnet"])
        ),
        "mean_delta_healthy": float(sub.loc[sub["y_true"] == 0, "delta"].mean()),
        "mean_delta_cancer": float(sub.loc[sub["y_true"] == 1, "delta"].mean()),
    }
    pd.DataFrame([summary]).to_csv(args.out_dir / f"{stem}_summary.csv", index=False)

    plot_paired_scatter(df, args.out_dir, args.dpi, args.stage_key, args.stage_label)
    plot_precision_at_k(df, args.out_dir, args.dpi, args.stage_key, args.stage_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
