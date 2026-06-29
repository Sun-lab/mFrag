#!/usr/bin/env python3
"""Grouped barplot: Stage II top-2 mFrag-Net, FP healthy, and both-low healthy."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

CFCRAFT = Path("/fh/working/sun_w/saishi_cui/cfCraft")
OUT_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure5"
PER_SAMPLE_CSV = OUT_DIR_DEFAULT / "figure5_frag_length_delfi_vs_mfragnet_stage_II_per_sample.csv"

COLOR_DELFI = "#1f77b4"
COLOR_MFRAGNET = "#9467bd"
LABEL_DELFI = "DELFI (S/L ratio)"
LABEL_MFRAGNET = "mFrag-Net (Frag length)"

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


def _short_id(sample_id: str) -> str:
    s = str(sample_id)
    return (
        s.replace("_lung_cancer", "")
        .replace("_healthy", "")
        .replace("_1", "")
        .replace("_2", "")
    )


def _select_samples(df: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    """Return (sample_id, short_id, group_label, selection) in plot order."""
    cancer = df.loc[(df["y_true"] == 1) & (df["stage_grp"] == "stage_II")].copy()
    top_cancer = cancer.nlargest(2, "predicted_proba_class1_mfragnet")

    healthy = df.loc[df["y_true"] == 0].copy()
    healthy["gap"] = healthy["predicted_proba_class1"] - healthy["predicted_proba_class1_mfragnet"]
    top_fp = healthy.nlargest(2, "gap")
    fp_ids = set(top_fp["sample_id"].astype(str))

    remaining = healthy.loc[~healthy["sample_id"].astype(str).isin(fp_ids)].copy()
    remaining["prob_sum"] = (
        remaining["predicted_proba_class1"] + remaining["predicted_proba_class1_mfragnet"]
    )
    low_both = remaining.nsmallest(2, "prob_sum")

    rows: list[tuple[str, str, str, str]] = []
    for sid in top_cancer["sample_id"].astype(str):
        rows.append((sid, _short_id(sid), "Stage II", "stage_II_top_mfragnet"))
    for sid in top_fp["sample_id"].astype(str):
        rows.append((sid, _short_id(sid), "Healthy", "healthy_fp"))
    for sid in low_both["sample_id"].astype(str):
        rows.append((sid, _short_id(sid), "Healthy", "healthy_both_low"))
    return rows


def _load_subset(df: pd.DataFrame, targets: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    ids = [sid for sid, _, _, _ in targets]
    sub = df.loc[df["sample_id"].astype(str).isin(ids)].copy()
    if len(sub) != len(targets):
        missing = set(ids) - set(sub["sample_id"].astype(str))
        raise SystemExit(f"Missing predictions for: {sorted(missing)}")
    order = {sid: i for i, (sid, _, _, _) in enumerate(targets)}
    sub["_ord"] = sub["sample_id"].astype(str).map(order)
    return sub.sort_values("_ord")


def plot_barplot(
    df: pd.DataFrame,
    targets: list[tuple[str, str, str, str]],
    out_dir: Path,
    dpi: int,
) -> None:
    n = len(targets)
    x = np.arange(n, dtype=float)
    delfi = df["predicted_proba_class1"].to_numpy(dtype=np.float64)
    mnet = df["predicted_proba_class1_mfragnet"].to_numpy(dtype=np.float64)
    tick_labels = [f"{short}\n({grp})" for _, short, grp, _ in targets]

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
        mnet,
        width=BAR_WIDTH,
        color=COLOR_MFRAGNET,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.92,
        label=LABEL_MFRAGNET,
        zorder=2,
    )

    ymax = float(max(delfi.max(), mnet.max()))
    ax.set_ylim(0, min(1.0, ymax * 1.12 + 0.04))
    ax.set_xticks(x, tick_labels, fontweight="bold")
    ax.set_ylabel("Predicted cancer probability", fontsize=FONT_LABEL, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

    legend_handles = [
        Patch(facecolor=COLOR_DELFI, edgecolor="black", linewidth=1.2, label=LABEL_DELFI),
        Patch(facecolor=COLOR_MFRAGNET, edgecolor="black", linewidth=1.2, label=LABEL_MFRAGNET),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.92, prop={"weight": "bold", "size": FONT_LEGEND})

    fig.tight_layout()
    stem = "figure5_frag_length_stage_II_mfragnet_top2_vs_fp_healthy_predicted_prob_barplot"
    for ext in ("png", "pdf"):
        out = out_dir / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=dpi)
        print(f"Wrote {out}")
    plt.close(fig)

    rows = []
    for (sid, short, grp, selection), pd_d, pm in zip(targets, delfi, mnet):
        rows.append(
            {
                "sample_id": sid,
                "short_id": short,
                "group": grp,
                "selection": selection,
                "prob_delfi": float(pd_d),
                "prob_mfrag_net": float(pm),
                "gap_delfi_minus_mfragnet": float(pd_d - pm),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / f"{stem}.csv", index=False)
    print(f"Wrote {out_dir / f'{stem}.csv'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    if not PER_SAMPLE_CSV.is_file():
        raise SystemExit(f"Missing {PER_SAMPLE_CSV}; run mfragnet diagnostic first.")

    _apply_style(dpi=args.dpi)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PER_SAMPLE_CSV)
    targets = _select_samples(df)
    sub = _load_subset(df, targets)
    plot_barplot(sub, targets, args.out_dir, args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
