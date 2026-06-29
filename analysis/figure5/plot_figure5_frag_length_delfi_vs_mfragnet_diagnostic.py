#!/usr/bin/env python3
"""Figure5: DELFI (S/L ratio) vs mFrag-Net (Frag length) paired scatter by stage."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

CFCRAFT = Path("/fh/working/sun_w/saishi_cui/cfCraft")
MATHIOS = CFCRAFT / "Mathios"
SL_DIR = MATHIOS / "Mathios_slratio_method"
FD_DIR = MATHIOS / "FDConv"
META_PATH = MATHIOS / "metadata" / "validation_testing_sample_list.tsv"
OUT_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure5"

DELFI_CSV = SL_DIR / "pca66_l1_baseline" / "test_predictions.csv"
MFRAGNET_NPZ = FD_DIR / "frag_meth3ch_new" / "runs" / "rep_031.npz"

COLOR_HEALTHY = "#2166AC"
COLOR_DELFI = "#1f77b4"
COLOR_MFRAGNET = "#9467bd"
STAGE_COLORS = {
    "I": "#d62728",
    "II": "#ff7f0e",
    "III": "#2ca02c",
    "IV": "#9467bd",
    "III + IV": "#2ca02c",
}
FONT_AXIS_LABEL = 13
ZOOM_X_LIM = (0.6, 1.0)
ZOOM_Y_LIM = (0.0, 1.0)
TOP_K = 10
SHADE_ALPHA = 0.18
PIECEWISE_BREAK = 0.99
PIECEWISE_HI = 1.0
PIECEWISE_HIGH_STRETCH = 20.0
AXIS_SCALE_DEFAULT = "piecewise_high"
X_TICK_CANDIDATES = np.array([0.6, 0.7, 0.8, 0.9, 0.99], dtype=np.float64)

STAGE_KEY_DEFAULT = "stage_II"
STAGE_LABEL_DEFAULT = "Stage II"
LEGEND_HEALTHY = "Healthy"
LEGEND_CANCER = {
    "stage_I": "Stage I lung cancer",
    "stage_II": "Stage II lung cancer",
}


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


def _stage_color(stage_label: str) -> str:
    key = stage_label.replace("Stage ", "").strip()
    return STAGE_COLORS.get(key, "#7f7f7f")


def _load_cnn_probs(npz_path: Path) -> pd.DataFrame:
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
    d = pd.read_csv(DELFI_CSV)
    m = _load_cnn_probs(MFRAGNET_NPZ)
    df = d.merge(m, on=["sample_id", "y_true"])
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
    return df


def subset_stage(df: pd.DataFrame, stage_key: str) -> pd.DataFrame:
    healthy = df["y_true"] == 0
    cancer = (df["y_true"] == 1) & (df["stage_grp"] == stage_key)
    return df[healthy | cancer].copy()


def _file_stem(stage_key: str) -> str:
    return f"figure5_frag_length_delfi_vs_mfragnet_{stage_key}"


def _top_k_threshold(prob: np.ndarray, k: int) -> float:
    if k <= 0 or len(prob) == 0:
        return np.nan
    k = min(k, len(prob))
    return float(np.sort(prob)[::-1][k - 1])


def _prob_to_x_axis(p: np.ndarray | float, x_lo: float) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, x_lo, PIECEWISE_HI)
    low_span = PIECEWISE_BREAK - x_lo
    return np.where(
        p <= PIECEWISE_BREAK,
        p - x_lo,
        low_span + PIECEWISE_HIGH_STRETCH * (p - PIECEWISE_BREAK),
    )


def _x_axis_limits(x_lo: float, axis_scale: str) -> tuple[float, float]:
    if axis_scale == "linear":
        return x_lo, ZOOM_X_LIM[1]
    hi_ax = (PIECEWISE_BREAK - x_lo) + PIECEWISE_HIGH_STRETCH * (PIECEWISE_HI - PIECEWISE_BREAK)
    return 0.0, hi_ax


def _format_prob_tick(p: float) -> str:
    return f"{p:.2f}".rstrip("0").rstrip(".")


def _apply_x_axis_ticks(ax, x_lo: float, x_hi: float, axis_scale: str) -> None:
    if axis_scale == "linear":
        return
    ticks = X_TICK_CANDIDATES[
        (X_TICK_CANDIDATES >= x_lo) & (X_TICK_CANDIDATES <= PIECEWISE_BREAK)
    ]
    if ticks.size == 0:
        ticks = np.array([x_lo, PIECEWISE_BREAK], dtype=np.float64)
    tick_pos = _prob_to_x_axis(ticks, x_lo)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([_format_prob_tick(float(p)) for p in ticks])


def _annotate_piecewise_x_axis(ax, x_lo: float) -> None:
    from matplotlib.transforms import blended_transform_factory

    break_ax = float(_prob_to_x_axis(PIECEWISE_BREAK, x_lo))
    end_ax = float(_prob_to_x_axis(PIECEWISE_HI, x_lo))
    mid_ax = 0.5 * (break_ax + end_ax)
    trans_x = blended_transform_factory(ax.transData, ax.transAxes)

    ax.axvline(break_ax, color="0.72", ls=":", lw=1.0, alpha=0.85, zorder=1)
    ax.plot(
        [break_ax, break_ax],
        [-0.018, 0.018],
        transform=trans_x,
        color="0.35",
        lw=1.3,
        clip_on=False,
        solid_capstyle="round",
    )
    ax.text(
        end_ax,
        -0.028,
        "1.0",
        transform=trans_x,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        clip_on=False,
    )
    ax.text(
        mid_ax,
        -0.075,
        rf"$\times {int(PIECEWISE_HIGH_STRETCH)}$",
        transform=trans_x,
        ha="center",
        va="top",
        fontsize=9,
        color="0.40",
        fontstyle="italic",
        clip_on=False,
    )
    ax.text(
        mid_ax,
        -0.11,
        f"{PIECEWISE_BREAK:.2f}–{PIECEWISE_HI:.1f} expanded",
        transform=trans_x,
        ha="center",
        va="top",
        fontsize=8,
        color="0.40",
        clip_on=False,
    )


def plot_paired_scatter(
    df: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    stage_key: str,
    stage_label: str,
    *,
    axis_scale: str = AXIS_SCALE_DEFAULT,
) -> None:
    sub = subset_stage(df, stage_key)
    x_lo, x_hi = ZOOM_X_LIM
    y_lo, y_hi = ZOOM_Y_LIM

    zoom = sub[
        (sub["predicted_proba_class1"] >= x_lo)
        & (sub["predicted_proba_class1"] <= x_hi)
        & (sub["predicted_proba_class1_mfragnet"] >= y_lo)
        & (sub["predicted_proba_class1_mfragnet"] <= y_hi)
    ].copy()
    h = zoom[zoom["y_true"] == 0]
    c = zoom[zoom["y_true"] == 1]
    c_color = _stage_color(stage_label)

    thr_delfi = _top_k_threshold(sub["predicted_proba_class1"].to_numpy(), TOP_K)
    thr_mfragnet = _top_k_threshold(zoom["predicted_proba_class1_mfragnet"].to_numpy(), TOP_K)
    print(
        f"Top-{TOP_K} DELFI threshold (global n={len(sub)}): x={thr_delfi:.4f}; "
        f"mFrag-Net threshold ({len(zoom)} visible): y={thr_mfragnet:.4f}"
    )

    x_lo_ax, x_hi_ax = _x_axis_limits(x_lo, axis_scale)
    if axis_scale == "linear":
        x_pad = 0.006
    else:
        x_pad = 0.02 * (x_hi_ax - x_lo_ax)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    if np.isfinite(thr_mfragnet):
        ax.axhspan(
            max(thr_mfragnet, y_lo),
            y_hi,
            xmin=0,
            xmax=1,
            color=COLOR_MFRAGNET,
            alpha=SHADE_ALPHA,
            zorder=0,
        )
        ax.axhline(thr_mfragnet, color=COLOR_MFRAGNET, lw=1.6, ls="--", zorder=1)
        text_x = (x_lo + x_pad) if axis_scale == "linear" else (x_lo_ax + x_pad)
        ax.text(
            text_x,
            thr_mfragnet + 0.006,
            f"predicted top {TOP_K}",
            color=COLOR_MFRAGNET,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
            zorder=5,
        )
    if np.isfinite(thr_delfi):
        thr_delfi_ax = float(_prob_to_x_axis(thr_delfi, x_lo)) if axis_scale == "piecewise_high" else thr_delfi
        ax.axvspan(
            max(thr_delfi_ax, x_lo_ax),
            x_hi_ax,
            ymin=0,
            ymax=1,
            color=COLOR_DELFI,
            alpha=SHADE_ALPHA,
            zorder=0,
        )
        ax.axvline(thr_delfi_ax, color=COLOR_DELFI, lw=1.6, ls="--", zorder=1)
        text_x = (thr_delfi - x_pad) if axis_scale == "linear" else (thr_delfi_ax - x_pad)
        ax.text(
            text_x,
            y_lo + 0.012,
            f"predicted top {TOP_K}",
            color=COLOR_DELFI,
            fontsize=9,
            fontweight="bold",
            ha="right",
            va="bottom",
            rotation=90,
            zorder=5,
        )

    if len(h):
        x_h = h["predicted_proba_class1"].to_numpy()
        if axis_scale == "piecewise_high":
            x_h = _prob_to_x_axis(x_h, x_lo)
        ax.scatter(
            x_h,
            h["predicted_proba_class1_mfragnet"],
            s=34,
            c=COLOR_HEALTHY,
            alpha=0.82,
            edgecolors="none",
            label=LEGEND_HEALTHY,
            zorder=3,
        )
    if len(c):
        x_c = c["predicted_proba_class1"].to_numpy()
        if axis_scale == "piecewise_high":
            x_c = _prob_to_x_axis(x_c, x_lo)
        ax.scatter(
            x_c,
            c["predicted_proba_class1_mfragnet"],
            s=42,
            c=c_color,
            alpha=0.9,
            edgecolors="none",
            label=LEGEND_CANCER[stage_key],
            zorder=4,
        )

    pr_d = average_precision_score(sub["y_true"], sub["predicted_proba_class1"])
    pr_m = average_precision_score(sub["y_true"], sub["predicted_proba_class1_mfragnet"])
    auc_d = roc_auc_score(sub["y_true"], sub["predicted_proba_class1"])
    auc_m = roc_auc_score(sub["y_true"], sub["predicted_proba_class1_mfragnet"])

    ax.set_xlim(x_lo_ax, x_hi_ax)
    ax.set_ylim(y_lo, y_hi)
    _apply_x_axis_ticks(ax, x_lo, x_hi, axis_scale)
    if axis_scale == "piecewise_high":
        _annotate_piecewise_x_axis(ax, x_lo)
    elif axis_scale == "linear":
        ax.set_box_aspect(1)
    ax.set_xlabel(
        "DELFI (S/L ratio)",
        fontsize=FONT_AXIS_LABEL,
        fontweight="bold",
        color="black",
    )
    ax.set_ylabel(
        "mFrag-Net (Frag length)",
        fontsize=FONT_AXIS_LABEL,
        fontweight="bold",
        color="black",
    )
    ax.set_title(
        f"{stage_label}: AUC {auc_d:.3f} → {auc_m:.3f}; PRAUC {pr_d:.3f} → {pr_m:.3f}",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9, prop={"weight": "bold", "size": 9})
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    if axis_scale == "piecewise_high":
        fig.subplots_adjust(bottom=0.16)
    else:
        fig.tight_layout()

    stem = _file_stem(stage_key)
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}_paired_scatter.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=dpi)
        print(f"Wrote {p}")
    plt.close(fig)


def _precision_at_top_k(y: np.ndarray, prob: np.ndarray, k: int) -> float:
    if k <= 0:
        return np.nan
    order = np.argsort(-prob)
    top = order[:k]
    return float(y[top].mean())


def plot_precision_at_k(df: pd.DataFrame, out_dir: Path, dpi: int, stage_key: str, stage_label: str) -> None:
    sub = subset_stage(df, stage_key)
    y = sub["y_true"].to_numpy(dtype=np.int32)
    p_d = sub["predicted_proba_class1"].to_numpy(dtype=np.float64)
    p_m = sub["predicted_proba_class1_mfragnet"].to_numpy(dtype=np.float64)
    n = len(y)
    n_healthy = int((sub["y_true"] == 0).sum())
    n_cancer = int((sub["y_true"] == 1).sum())
    ks = np.unique(np.round(np.linspace(1, n, 80)).astype(int))
    prec_d = [_precision_at_top_k(y, p_d, k) for k in ks]
    prec_m = [_precision_at_top_k(y, p_m, k) for k in ks]
    prev = float(y.mean())

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(ks, prec_d, color=COLOR_DELFI, lw=2.2, label="DELFI (S/L ratio)")
    ax.plot(ks, prec_m, color=COLOR_MFRAGNET, lw=2.2, label="mFrag-Net (Frag length)")
    ax.axhline(prev, color="k", ls="--", lw=1, alpha=0.4, label=f"prevalence = {prev:.3f}")
    ax.set_xlabel("Top k samples flagged")
    ax.set_ylabel("Precision among flagged")
    ax.set_title(
        f"{stage_label}: precision @ top-k (healthy = {n_healthy}, cancer = {n_cancer})",
        fontweight="bold",
    )
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.9, prop={"weight": "bold"})
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    fig.tight_layout()

    stem = _file_stem(stage_key)
    rows = []
    for k, fd, fm in zip(ks, prec_d, prec_m):
        rows.append({"k": int(k), "fraction": k / n, "precision_delfi": fd, "precision_mfrag_net": fm})
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
    ap.add_argument("--stage-key", type=str, default=STAGE_KEY_DEFAULT)
    ap.add_argument("--stage-label", type=str, default=STAGE_LABEL_DEFAULT)
    ap.add_argument(
        "--axis-scale",
        choices=("linear", "piecewise_high"),
        default=AXIS_SCALE_DEFAULT,
        help="x-axis scale (default: piecewise_high stretches 0.99-1.0 by 10x; y-axis stays linear)",
    )
    args = ap.parse_args()

    _apply_style(dpi=args.dpi)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_merged()
    sub = subset_stage(df, args.stage_key)
    stem = _file_stem(args.stage_key)
    sub.to_csv(args.out_dir / f"{stem}_per_sample.csv", index=False)

    summary = {
        "stage_key": args.stage_key,
        "stage_label": args.stage_label,
        "n": len(sub),
        "n_healthy": int((sub["y_true"] == 0).sum()),
        "n_cancer": int((sub["y_true"] == 1).sum()),
        "auc_delfi": float(roc_auc_score(sub["y_true"], sub["predicted_proba_class1"])),
        "auc_mfrag_net": float(roc_auc_score(sub["y_true"], sub["predicted_proba_class1_mfragnet"])),
        "prauc_delfi": float(average_precision_score(sub["y_true"], sub["predicted_proba_class1"])),
        "prauc_mfrag_net": float(
            average_precision_score(sub["y_true"], sub["predicted_proba_class1_mfragnet"])
        ),
    }
    pd.DataFrame([summary]).to_csv(args.out_dir / f"{stem}_summary.csv", index=False)

    plot_paired_scatter(
        df,
        args.out_dir,
        args.dpi,
        args.stage_key,
        args.stage_label,
        axis_scale=args.axis_scale,
    )
    plot_precision_at_k(df, args.out_dir, args.dpi, args.stage_key, args.stage_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
