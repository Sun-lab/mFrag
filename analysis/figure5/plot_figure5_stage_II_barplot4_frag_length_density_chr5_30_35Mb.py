#!/usr/bin/env python3
"""Fragment-length density for barplot-6 samples, restricted to Chr5:30–35 Mb."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_frag = Path(__file__).resolve().parents[2] / "Mathios" / "Fragment_length"
if str(_frag) not in sys.path:
    sys.path.insert(0, str(_frag))

from plot_fragment_length_density_overlay import (  # noqa: E402
    CHUNK,
    LEN_MAX,
    LEN_MIN,
    N_LEN,
    _draw_zebra_bins,
    _set_binned_xticks,
    _slice_renorm,
    _smooth_curve,
)

CFCRAFT = Path("/fh/working/sun_w/saishi_cui/cfCraft")
MATHIOS = CFCRAFT / "Mathios"
TEST_FRAG = MATHIOS / "fragments_csv" / "testing"
OUT_DIR = CFCRAFT / "mFrag_Figures" / "Figure5"
BARPLOT_CSV = OUT_DIR / "figure5_frag_length_stage_II_mfragnet_top2_vs_fp_healthy_predicted_prob_barplot.csv"

PLOT_MIN = 100
PLOT_MAX = 220
BIN_WIDTH = 5
LINE_LW = 2.8
LINE_ALPHA = 0.92

REGION_CHROM = "chr5"
REGION_START = 30_000_000
REGION_END = 35_000_000

SELECTION_COLOR = {
    "stage_II_top_mfragnet": "#ff7f0e",
    "healthy_fp": "#2166AC",
    "healthy_both_low": "#2ca02c",
}

SELECTION_LABEL = {
    "stage_II_top_mfragnet": "Stage II",
    "healthy_fp": "Healthy",
    "healthy_both_low": "Healthy",
}


def _curve(dens: np.ndarray) -> np.ndarray:
    return _smooth_curve(_slice_renorm(dens, PLOT_MIN, PLOT_MAX))


def _sample_weighted_density_region(path_str: str) -> tuple[str, str, np.ndarray]:
    path = Path(path_str)
    hist = np.zeros(N_LEN, dtype=np.float64)

    usecols = ["chromosome", "start", "fragment_length", "weight"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK):
        df = chunk.rename(columns={"chromosome": "chrom"})
        df["chrom"] = df["chrom"].astype(str)
        df["chrom"] = df["chrom"].where(
            df["chrom"].str.startswith("chr"), "chr" + df["chrom"]
        )
        df = df[df["chrom"] == REGION_CHROM]
        df = df[(df["start"] >= REGION_START) & (df["start"] < REGION_END)]
        if df.empty:
            continue

        fl = df["fragment_length"].to_numpy(dtype=np.int32, copy=False)
        w = df["weight"].to_numpy(dtype=np.float64, copy=False)
        ok = (fl >= LEN_MIN) & (fl <= LEN_MAX)
        if not ok.any():
            continue
        idx = (fl[ok] - LEN_MIN).astype(np.intp, copy=False)
        np.add.at(hist, idx, w[ok])

    total = hist.sum()
    if total > 0:
        hist /= total

    group = "healthy" if path.name.endswith("_healthy.csv") else "lung_cancer"
    return path.stem, group, hist


def _load_targets() -> list[tuple[str, str, str, str]]:
    df = pd.read_csv(BARPLOT_CSV)
    rows: list[tuple[str, str, str, str]] = []
    for _, r in df.iterrows():
        rows.append(
            (
                str(r["sample_id"]),
                str(r["short_id"]),
                str(r["group"]),
                str(r["selection"]),
            )
        )
    if len(rows) != 6:
        raise SystemExit(f"Expected 6 barplot samples, found {len(rows)}")
    return rows


def plot_density(
    curves: list[tuple[str, str, str, str, np.ndarray]],
    *,
    out_png: Path,
    out_pdf: Path,
    dpi: int,
) -> None:
    x = np.arange(PLOT_MIN, PLOT_MAX + 1, dtype=np.float64)

    plt.rcParams.update(
        {
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
        }
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(PLOT_MIN, PLOT_MAX)
    _draw_zebra_bins(ax, PLOT_MIN, PLOT_MAX, BIN_WIDTH)

    for _, short, grp, selection, y in curves:
        color = SELECTION_COLOR[selection]
        grp_label = SELECTION_LABEL[selection]
        ax.plot(
            x,
            y,
            color=color,
            linewidth=LINE_LW,
            alpha=LINE_ALPHA,
            label=f"{short} ({grp_label})",
            zorder=3,
        )

    ax.set_xlabel("Fragment length (bp)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Weighted density", fontsize=13, fontweight="bold")
    ax.set_title(
        "Fragment length density (100–220 bp): Chr5:30–35 Mb\nbarplot-selected samples",
        fontsize=14,
        fontweight="bold",
    )
    _set_binned_xticks(ax, PLOT_MIN, PLOT_MAX, BIN_WIDTH)
    ax.legend(loc="upper right", frameon=True, fontsize=10, prop={"weight": "bold"})
    ax.grid(True, alpha=0.25, axis="y")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

    fig.subplots_adjust(bottom=0.18)
    fig.tight_layout()
    for out in (out_png, out_pdf):
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    if not BARPLOT_CSV.is_file():
        raise SystemExit(f"Missing {BARPLOT_CSV}; run barplot script first.")

    targets = _load_targets()
    curves: list[tuple[str, str, str, str, np.ndarray]] = []
    for sid, short, grp, selection in targets:
        path = TEST_FRAG / f"{sid}.csv"
        if not path.is_file():
            raise SystemExit(f"Missing fragment CSV: {path}")
        _, _, dens = _sample_weighted_density_region(str(path))
        curves.append((sid, short, grp, selection, _curve(dens)))

    stem = "figure5_stage_II_barplot4_frag_length_density_chr5_30_35Mb"
    plot_density(
        curves,
        out_png=args.out_dir / f"{stem}.png",
        out_pdf=args.out_dir / f"{stem}.pdf",
        dpi=args.dpi,
    )

    rows = [
        {"sample_id": sid, "short_id": short, "group": grp, "selection": selection}
        for sid, short, grp, selection, _ in curves
    ]
    pd.DataFrame(rows).to_csv(args.out_dir / f"{stem}_samples.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
