#!/usr/bin/env python3
"""Grouped barplot: FP healthy + both-low healthy, DELFI vs mFrag-Weight."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

CFCRAFT = Path("/fh/working/sun_w/saishi_cui/cfCraft")
MATHIOS = CFCRAFT / "Mathios"
SL_DIR = MATHIOS / "Mathios_slratio_method"
OUT_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure5"
PER_SAMPLE_CSV = OUT_DIR_DEFAULT / "figure5_frag_length_delfi_vs_mfragdelfi_stage_I_per_sample.csv"

DELFI_CSV = SL_DIR / "pca66_l1_baseline" / "test_predictions.csv"
MFRAG_CSV = SL_DIR / "pca66_l1_coupling_abs_dbeta_over_mean" / "test_predictions.csv"

FP_SAMPLES = (
    ("CGPLH676P_1_healthy", "CGPLH676P_1"),
    ("CGPLH969P_healthy", "CGPLH969P"),
    ("CGPLH919P_healthy", "CGPLH919P"),
)

COLOR_DELFI = "#1f77b4"
COLOR_MFRAG = "#ff7f0e"
LABEL_DELFI = "DELFI (S/L ratio)"
LABEL_MFRAG = "mFrag-Weight (S/L ratio)"

FONT_LABEL = 14
FONT_TICK = 12
FONT_LEGEND = 14
BAR_WIDTH = 0.34


def _apply_style(*, dpi: int) -> None:
    mpl.rcParams.update(
        {
            "font.size": FONT_TICK,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _select_both_low(df: pd.DataFrame, n: int = 3) -> list[tuple[str, str]]:
    fp_ids = {sid for sid, _ in FP_SAMPLES}
    healthy = df.loc[df["y_true"] == 0].copy()
    healthy["prob_sum"] = (
        healthy["predicted_proba_class1_delfi"] + healthy["predicted_proba_class1_mfrag"]
    )
    sub = healthy.loc[~healthy["sample_id"].astype(str).isin(fp_ids)]
    top = sub.nsmallest(n, "prob_sum")
    out: list[tuple[str, str]] = []
    for _, r in top.iterrows():
        sid = str(r["sample_id"])
        short = str(r.get("short_id", sid.replace("_healthy", "")))
        out.append((sid, short))
    if len(out) != n:
        raise SystemExit(f"Expected {n} both-low healthy samples, found {len(out)}")
    return out


def _select_targets(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Return (sample_id, short_id, selection) in plot order."""
    rows: list[tuple[str, str, str]] = []
    for sid, short in FP_SAMPLES:
        rows.append((sid, short, "healthy_fp"))
    for sid, short in _select_both_low(df):
        rows.append((sid, short, "healthy_both_low"))
    return rows


def _load_subset(df: pd.DataFrame, targets: list[tuple[str, str, str]]) -> pd.DataFrame:
    ids = [sid for sid, _, _ in targets]
    sub = df.loc[df["sample_id"].astype(str).isin(ids)].copy()
    if len(sub) != len(targets):
        missing = set(ids) - set(sub["sample_id"].astype(str))
        raise SystemExit(f"Missing predictions for: {sorted(missing)}")
    order = {sid: i for i, (sid, _, _) in enumerate(targets)}
    sub["_ord"] = sub["sample_id"].astype(str).map(order)
    return sub.sort_values("_ord")


def _load_merged() -> pd.DataFrame:
    if PER_SAMPLE_CSV.is_file():
        return pd.read_csv(PER_SAMPLE_CSV)
    d = pd.read_csv(DELFI_CSV)
    m = pd.read_csv(MFRAG_CSV)
    return d.merge(m, on="sample_id", suffixes=("_delfi", "_mfrag"))


def plot_barplot(
    df: pd.DataFrame,
    targets: list[tuple[str, str, str]],
    out_dir: Path,
    dpi: int,
) -> None:
    n = len(targets)
    x = np.arange(n, dtype=float)
    delfi = df["predicted_proba_class1_delfi"].to_numpy(dtype=np.float64)
    mfrag = df["predicted_proba_class1_mfrag"].to_numpy(dtype=np.float64)
    tick_labels = [f"{short}\n(Healthy)" for _, short, _ in targets]

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.bar(
        x - BAR_WIDTH / 2,
        delfi,
        width=BAR_WIDTH,
        color=COLOR_DELFI,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.92,
        label=LABEL_DELFI,
        zorder=2,
    )
    ax.bar(
        x + BAR_WIDTH / 2,
        mfrag,
        width=BAR_WIDTH,
        color=COLOR_MFRAG,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.92,
        label=LABEL_MFRAG,
        zorder=2,
    )

    ymax = float(max(delfi.max(), mfrag.max()))
    ax.set_ylim(0, min(1.0, ymax * 1.12 + 0.04))
    ax.set_xticks(x, tick_labels, fontweight="bold")
    ax.set_ylabel("Predicted cancer probability", fontsize=FONT_LABEL, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

    legend_handles = [
        Patch(facecolor=COLOR_DELFI, edgecolor="black", linewidth=1.2, label=LABEL_DELFI),
        Patch(facecolor=COLOR_MFRAG, edgecolor="black", linewidth=1.2, label=LABEL_MFRAG),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.92, prop={"weight": "bold", "size": FONT_LEGEND})

    fig.tight_layout()
    stem = "figure5_frag_length_top3_fp_healthy_predicted_prob_barplot"
    for ext in ("png", "pdf"):
        out = out_dir / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=dpi)
        print(f"Wrote {out}")
    plt.close(fig)

    rows = []
    for (sid, short, selection), pd_d, pm in zip(targets, delfi, mfrag):
        rows.append(
            {
                "sample_id": sid,
                "short_id": short,
                "selection": selection,
                "prob_delfi": float(pd_d),
                "prob_mfrag_delfi": float(pm),
                "gap_delfi_minus_mfragdelfi": float(pd_d - pm),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / f"{stem}.csv", index=False)
    print(f"Wrote {out_dir / f'{stem}.csv'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    _apply_style(dpi=args.dpi)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_merged()
    targets = _select_targets(df)
    sub = _load_subset(df, targets)
    plot_barplot(sub, targets, args.out_dir, args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
