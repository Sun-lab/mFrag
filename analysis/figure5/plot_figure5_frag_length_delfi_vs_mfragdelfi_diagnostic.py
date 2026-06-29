#!/usr/bin/env python3
"""
Figure5: DELFI (Frag length) vs mFrag-Weight diagnostic plots.

Default: Stage I cancer + all healthy controls.
Outputs: paired scatter + precision@top-k (no forest / delta bars).
"""
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
META_PATH = MATHIOS / "metadata" / "validation_testing_sample_list.tsv"
OUT_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure5"

DELFI_CSV = SL_DIR / "pca66_l1_baseline" / "test_predictions.csv"
MFRAG_CSV = SL_DIR / "pca66_l1_coupling_abs_dbeta_over_mean" / "test_predictions.csv"

COLOR_HEALTHY = "#2166AC"
COLOR_DELFI = "#1f77b4"
COLOR_MFRAG = "#ff7f0e"
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
LEGEND_HEALTHY = "Healthy"
LEGEND_CANCER = {
    "stage_I": "Stage I lung cancer",
    "stage_II": "Stage II lung cancer",
}
FONT_AXIS_LABEL = 13
ZOOM_LIM = (0.8, 1.0)
TOP_K = 10
SHADE_ALPHA = 0.18
PIECEWISE_LO = 0.8
PIECEWISE_BREAK = 0.99
PIECEWISE_HI = 1.0
PIECEWISE_HIGH_STRETCH = 10.0
PIECEWISE_LOW_SPAN = PIECEWISE_BREAK - PIECEWISE_LO
PIECEWISE_HIGH_SPAN = PIECEWISE_HI - PIECEWISE_BREAK
AXIS_SCALE_DEFAULT = "piecewise_high"
PROB_TICK_CANDIDATES = np.array([0.80, 0.85, 0.90, 0.95, 0.99], dtype=np.float64)


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


def load_merged() -> pd.DataFrame:
    d = pd.read_csv(DELFI_CSV)
    m = pd.read_csv(MFRAG_CSV)
    df = d.merge(m, on=["sample_id", "y_true"], suffixes=("_delfi", "_mfrag"))
    df["delta"] = df["predicted_proba_class1_mfrag"] - df["predicted_proba_class1_delfi"]
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


def _file_stem(stage_key: str) -> str:
    return f"figure5_frag_length_delfi_vs_mfragdelfi_{stage_key}"


def _top_k_threshold(prob: np.ndarray, k: int) -> float:
    if k <= 0 or len(prob) == 0:
        return np.nan
    k = min(k, len(prob))
    return float(np.sort(prob)[::-1][k - 1])


def _prob_to_axis(p: np.ndarray | float, axis_scale: str) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if axis_scale == "linear":
        return p
    if axis_scale == "piecewise_high":
        p = np.clip(p, PIECEWISE_LO, PIECEWISE_HI)
        return np.where(
            p <= PIECEWISE_BREAK,
            p - PIECEWISE_LO,
            PIECEWISE_LOW_SPAN + PIECEWISE_HIGH_STRETCH * (p - PIECEWISE_BREAK),
        )
    raise ValueError(f"Unknown axis_scale: {axis_scale}")


def _axis_limits(axis_scale: str) -> tuple[float, float]:
    if axis_scale == "linear":
        return ZOOM_LIM
    hi_ax = PIECEWISE_LOW_SPAN + PIECEWISE_HIGH_STRETCH * PIECEWISE_HIGH_SPAN
    return 0.0, hi_ax


def _axis_prob_ticks(lo: float, hi: float, axis_scale: str) -> np.ndarray:
    candidates = PROB_TICK_CANDIDATES
    if axis_scale == "piecewise_high":
        candidates = PROB_TICK_CANDIDATES[PROB_TICK_CANDIDATES <= PIECEWISE_BREAK]
    ticks = candidates[(candidates >= lo) & (candidates <= hi)]
    if ticks.size == 0:
        ticks = np.array([lo, hi], dtype=np.float64)
    return ticks


def _format_prob_tick(p: float) -> str:
    return f"{p:.2f}".rstrip("0").rstrip(".")


def _apply_prob_axis_ticks(ax, axis: str, lo: float, hi: float, axis_scale: str) -> None:
    if axis_scale == "linear":
        return
    ticks = _axis_prob_ticks(lo, hi, axis_scale)
    tick_pos = _prob_to_axis(ticks, axis_scale)
    tick_lbl = [_format_prob_tick(float(p)) for p in ticks]
    if axis == "x":
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl)
    else:
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_lbl)


def _annotate_piecewise_axes(ax) -> None:
    """Mark the stretched 0.99–1.0 segment on the x-axis (no interior ticks)."""
    from matplotlib.transforms import blended_transform_factory

    break_ax = float(_prob_to_axis(PIECEWISE_BREAK, "piecewise_high"))
    end_ax = float(_prob_to_axis(PIECEWISE_HI, "piecewise_high"))
    mid_ax = 0.5 * (break_ax + end_ax)
    trans_x = blended_transform_factory(ax.transData, ax.transAxes)

    ax.axvline(break_ax, color="0.72", ls=":", lw=1.0, alpha=0.85, zorder=1)
    ax.axhline(break_ax, color="0.72", ls=":", lw=1.0, alpha=0.85, zorder=1)

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
    x_lo, x_hi = ZOOM_LIM
    y_lo, y_hi = ZOOM_LIM

    zoom = sub[
        (sub["predicted_proba_class1_delfi"] >= x_lo)
        & (sub["predicted_proba_class1_delfi"] <= x_hi)
        & (sub["predicted_proba_class1_mfrag"] >= y_lo)
        & (sub["predicted_proba_class1_mfrag"] <= y_hi)
    ].copy()
    h = zoom[zoom["y_true"] == 0]
    c = zoom[zoom["y_true"] == 1]

    thr_delfi = _top_k_threshold(sub["predicted_proba_class1_delfi"].to_numpy(), TOP_K)
    thr_mfrag = _top_k_threshold(sub["predicted_proba_class1_mfrag"].to_numpy(), TOP_K)
    print(f"Top-{TOP_K} thresholds: DELFI x={thr_delfi:.4f}, mFrag-Weight y={thr_mfrag:.4f}")

    x_lo_ax, x_hi_ax = _axis_limits(axis_scale)
    y_lo_ax, y_hi_ax = _axis_limits(axis_scale)
    if axis_scale == "linear":
        x_pad, y_pad = 0.006, 0.006
    else:
        x_pad = 0.02 * (x_hi_ax - x_lo_ax)
        y_pad = 0.02 * (y_hi_ax - y_lo_ax)

    fig, ax = plt.subplots(figsize=(6.5, 6.2))

    if np.isfinite(thr_mfrag):
        thr_mfrag_ax = float(_prob_to_axis(thr_mfrag, axis_scale))
        ax.axhspan(
            max(thr_mfrag_ax, y_lo_ax),
            y_hi_ax,
            xmin=0,
            xmax=1,
            color=COLOR_MFRAG,
            alpha=SHADE_ALPHA,
            zorder=0,
        )
        ax.axhline(thr_mfrag_ax, color=COLOR_MFRAG, lw=1.6, ls="--", zorder=1)
        text_x = (x_lo + x_pad) if axis_scale == "linear" else (x_lo_ax + x_pad)
        text_y = (thr_mfrag + y_pad) if axis_scale == "linear" else (thr_mfrag_ax + y_pad)
        ax.text(
            text_x,
            text_y,
            f"predicted top {TOP_K}",
            color=COLOR_MFRAG,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
            zorder=5,
        )
    if np.isfinite(thr_delfi):
        thr_delfi_ax = float(_prob_to_axis(thr_delfi, axis_scale))
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
        text_y = (y_lo + 2.0 * y_pad) if axis_scale == "linear" else (y_lo_ax + 2.0 * y_pad)
        ax.text(
            text_x,
            text_y,
            f"predicted top {TOP_K}",
            color=COLOR_DELFI,
            fontsize=9,
            fontweight="bold",
            ha="right",
            va="bottom",
            rotation=90,
            zorder=5,
        )

    diag_lo = max(x_lo, y_lo)
    diag_hi = min(x_hi, y_hi)
    if diag_hi > diag_lo:
        p_diag = np.linspace(diag_lo, diag_hi, 200, dtype=np.float64)
        ax.plot(
            _prob_to_axis(p_diag, axis_scale),
            _prob_to_axis(p_diag, axis_scale),
            "k--",
            lw=1,
            alpha=0.35,
            zorder=1,
        )

    if len(h):
        ax.scatter(
            _prob_to_axis(h["predicted_proba_class1_delfi"].to_numpy(), axis_scale),
            _prob_to_axis(h["predicted_proba_class1_mfrag"].to_numpy(), axis_scale),
            s=34,
            c=COLOR_HEALTHY,
            alpha=0.82,
            edgecolors="none",
            label=LEGEND_HEALTHY,
            zorder=3,
        )
    if len(c):
        stage_color = STAGE_COLORS.get(stage_key.replace("stage_", ""), STAGE_COLORS["I"])
        ax.scatter(
            _prob_to_axis(c["predicted_proba_class1_delfi"].to_numpy(), axis_scale),
            _prob_to_axis(c["predicted_proba_class1_mfrag"].to_numpy(), axis_scale),
            s=42,
            c=stage_color,
            alpha=0.9,
            edgecolors="none",
            label=LEGEND_CANCER[stage_key],
            zorder=4,
        )

    pr_d = average_precision_score(sub["y_true"], sub["predicted_proba_class1_delfi"])
    pr_m = average_precision_score(sub["y_true"], sub["predicted_proba_class1_mfrag"])
    auc_d = roc_auc_score(sub["y_true"], sub["predicted_proba_class1_delfi"])
    auc_m = roc_auc_score(sub["y_true"], sub["predicted_proba_class1_mfrag"])

    ax.set_xlim(x_lo_ax, x_hi_ax)
    ax.set_ylim(y_lo_ax, y_hi_ax)
    ax.set_aspect("equal")
    _apply_prob_axis_ticks(ax, "x", x_lo, x_hi, axis_scale)
    _apply_prob_axis_ticks(ax, "y", y_lo, y_hi, axis_scale)
    if axis_scale == "piecewise_high":
        _annotate_piecewise_axes(ax)
    ax.set_xlabel(
        "DELFI (S/L ratio)",
        fontsize=FONT_AXIS_LABEL,
        fontweight="bold",
        color="black",
    )
    ax.set_ylabel(
        "mFrag-Weight (S/L ratio)",
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
        fig.subplots_adjust(left=0.12, bottom=0.16)
    else:
        fig.tight_layout()

    stem = _file_stem(stage_key)
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}_paired_scatter.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=dpi)
        print(f"Wrote {p}")
    plt.close(fig)


def _sample_color(row: pd.Series) -> str:
    if row["y_true"] == 0:
        return COLOR_HEALTHY
    stage = str(row["Stage"]).strip().upper() if pd.notna(row["Stage"]) else ""
    return STAGE_COLORS.get(stage, "#7f7f7f")


def _short_id(sample_id: str) -> str:
    return sample_id.replace("_healthy", "").replace("_lung_cancer", "")


def plot_dual_forest(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    from matplotlib.patches import Patch

    work = df.copy()
    work["point_color"] = work.apply(_sample_color, axis=1)
    healthy = work[work["y_true"] == 0].sort_values("predicted_proba_class1_mfrag")
    cancer = work[work["y_true"] == 1].sort_values("predicted_proba_class1_mfrag")
    plot_df = pd.concat([healthy, cancer], ignore_index=True)
    n_h, n_c = len(healthy), len(cancer)
    gap = 2.5
    y_h = np.arange(n_h, dtype=float)
    y_c = np.arange(n_c, dtype=float) + n_h + gap
    y_all = np.concatenate([y_h, y_c])
    colors = plot_df["point_color"].to_numpy()
    labels = [_short_id(s) for s in plot_df["sample_id"]]
    split_y = n_h + gap * 0.5 - 0.5

    fig_h = max(12, 0.16 * len(plot_df) + 2)
    fig, (ax_m, ax_d) = plt.subplots(1, 2, figsize=(16, fig_h), sharey=True)

    for ax, prob_col, marker, title in (
        (ax_m, "predicted_proba_class1_mfrag", "o", "mFrag-Weight (S/L ratio)"),
        (ax_d, "predicted_proba_class1_delfi", "D", "DELFI (Frag length)"),
    ):
        ax.scatter(
            plot_df[prob_col],
            y_all,
            s=30,
            c=colors,
            alpha=0.9,
            marker=marker,
            edgecolors="0.25",
            linewidths=0.4,
            zorder=3,
        )
        if n_h and n_c:
            ax.axhline(split_y, color="black", lw=2.8, ls="--", zorder=1)
        ax.set_xlabel("Predicted probability (class 1 = lung cancer)", fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlim(-0.02, 1.05)
        ax.tick_params(axis="x", labelsize=12, width=1.2)
        ax.grid(True, axis="x", alpha=0.25)
        for lbl in ax.get_xticklabels():
            lbl.set_fontweight("bold")

    ax_m.set_yticks(y_all)
    ax_m.set_yticklabels(labels, fontsize=8.5, fontweight="bold")
    ax_m.set_ylabel("")
    plt.setp(ax_d.get_yticklabels(), visible=False)

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
    ax_m.legend(
        handles=stage_handles,
        loc="upper left",
        framealpha=0.92,
        prop={"weight": "bold", "size": 10},
        title="Sample group",
        title_fontproperties={"weight": "bold", "size": 10},
    )
    fig.tight_layout(rect=[0.10, 0.02, 0.98, 0.98])
    fig.subplots_adjust(wspace=0.08)
    for ext in ("png", "pdf"):
        p = out_dir / f"figure5_frag_length_delfi_vs_mfragdelfi_dual_forest.{ext}"
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
    p_d = sub["predicted_proba_class1_delfi"].to_numpy(dtype=np.float64)
    p_m = sub["predicted_proba_class1_mfrag"].to_numpy(dtype=np.float64)
    n = len(y)
    n_healthy = int((sub["y_true"] == 0).sum())
    n_cancer = int((sub["y_true"] == 1).sum())
    ks = np.unique(np.round(np.linspace(1, n, 80)).astype(int))
    prec_d = [_precision_at_top_k(y, p_d, k) for k in ks]
    prec_m = [_precision_at_top_k(y, p_m, k) for k in ks]
    prev = float(y.mean())

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(ks, prec_d, color=COLOR_DELFI, lw=2.2, label="DELFI (S/L ratio)")
    ax.plot(ks, prec_m, color=COLOR_MFRAG, lw=2.2, label="mFrag-Weight (S/L ratio)")
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
        rows.append({"k": int(k), "fraction": k / n, "precision_delfi": fd, "precision_mfrag_delfi": fm})
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
    ap.add_argument("--stage-key", type=str, default=STAGE_KEY, help="e.g. stage_I")
    ap.add_argument("--stage-label", type=str, default=STAGE_LABEL, help="e.g. Stage I")
    ap.add_argument(
        "--dual-forest",
        action="store_true",
        help="Also plot full-testing dual forest (all stages colored)",
    )
    ap.add_argument("--skip-stage-plots", action="store_true", help="Skip Stage-I scatter / precision@k")
    ap.add_argument(
        "--axis-scale",
        choices=("linear", "piecewise_high"),
        default=AXIS_SCALE_DEFAULT,
        help="Paired-scatter axis scale (default: piecewise_high stretches 0.99-1.0 by 10x)",
    )
    args = ap.parse_args()

    _apply_style(dpi=args.dpi)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_merged()
    df.to_csv(args.out_dir / "figure5_frag_length_delfi_vs_mfragdelfi_per_sample.csv", index=False)

    if args.dual_forest:
        plot_dual_forest(df, args.out_dir, args.dpi)

    if not args.skip_stage_plots:
        sub = subset_stage(df, args.stage_key)
        stem = _file_stem(args.stage_key)
        sub.to_csv(args.out_dir / f"{stem}_per_sample.csv", index=False)

        summary = {
            "stage_key": args.stage_key,
            "stage_label": args.stage_label,
            "n": len(sub),
            "n_healthy": int((sub["y_true"] == 0).sum()),
            "n_cancer": int((sub["y_true"] == 1).sum()),
            "auc_delfi": float(roc_auc_score(sub["y_true"], sub["predicted_proba_class1_delfi"])),
            "auc_mfrag_delfi": float(roc_auc_score(sub["y_true"], sub["predicted_proba_class1_mfrag"])),
            "prauc_delfi": float(average_precision_score(sub["y_true"], sub["predicted_proba_class1_delfi"])),
            "prauc_mfrag_delfi": float(
                average_precision_score(sub["y_true"], sub["predicted_proba_class1_mfrag"])
            ),
            "mean_delta_healthy": float(sub.loc[sub["y_true"] == 0, "delta"].mean()),
            "mean_delta_cancer": float(sub.loc[sub["y_true"] == 1, "delta"].mean()),
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
