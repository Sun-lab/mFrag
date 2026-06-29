#!/usr/bin/env python3
"""Stage II cancer vs testing healthy median fragment-length density (100–220 bp)."""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_frag = Path(__file__).resolve().parents[2] / "Mathios" / "Fragment_length"
if str(_frag) not in sys.path:
    sys.path.insert(0, str(_frag))

from plot_fragment_length_density_overlay import (  # noqa: E402
    COLOR_HEALTHY,
    _draw_zebra_bins,
    _sample_weighted_density,
    _set_binned_xticks,
    _slice_renorm,
    _smooth_curve,
)

CFCRAFT = Path("/fh/working/sun_w/saishi_cui/cfCraft")
MATHIOS = CFCRAFT / "Mathios"
TEST_FRAG = MATHIOS / "fragments_csv" / "testing"
OUT_DIR = CFCRAFT / "mFrag_Figures" / "Figure5"
PER_SAMPLE_CSV = OUT_DIR / "figure5_frag_length_delfi_vs_mfragnet_stage_II_per_sample.csv"

PLOT_MIN = 100
PLOT_MAX = 220
BIN_WIDTH = 5
COLOR_STAGE_II = "#ff7f0e"
HEALTHY_MEDIAN_LW = 3.2
CANCER_LW = 0.9
CANCER_ALPHA = 0.75


def _stage_ii_cancer_ids() -> list[str]:
    df = pd.read_csv(PER_SAMPLE_CSV)
    sub = df.loc[(df["y_true"] == 1) & (df["stage_grp"] == "stage_II"), "sample_id"]
    ids = sub.astype(str).tolist()
    if len(ids) != 12:
        raise SystemExit(f"Expected 12 Stage II cancer samples, found {len(ids)}")
    return ids


def _healthy_testing_paths() -> list[Path]:
    return sorted(TEST_FRAG.glob("*_healthy.csv"))


def _load_densities(paths: list[Path], workers: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_sample_weighted_density, str(p)): p for p in paths}
        for fut in as_completed(futs):
            _, _, dens = fut.result()
            out.append(dens)
    return out


def _curve(dens: np.ndarray) -> np.ndarray:
    return _smooth_curve(_slice_renorm(dens, PLOT_MIN, PLOT_MAX))


def plot_density(
    healthy_median: np.ndarray,
    cancer_curves: list[np.ndarray],
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

    for y in cancer_curves:
        ax.plot(x, y, color=COLOR_STAGE_II, alpha=CANCER_ALPHA, linewidth=CANCER_LW, zorder=2)

    ax.plot(
        x,
        healthy_median,
        color=COLOR_HEALTHY,
        linewidth=HEALTHY_MEDIAN_LW,
        label=f"Testing healthy (median, N={len(_healthy_testing_paths())})",
        zorder=4,
    )
    ax.plot(
        [],
        [],
        color=COLOR_STAGE_II,
        linewidth=CANCER_LW,
        label=f"Stage II lung cancer (N={len(cancer_curves)})",
    )

    ax.set_xlabel("Fragment length (bp)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Weighted density", fontsize=13, fontweight="bold")
    ax.set_title(
        "Fragment length density (100–220 bp): Stage II cancer vs testing healthy (median)",
        fontsize=14,
        fontweight="bold",
    )
    _set_binned_xticks(ax, PLOT_MIN, PLOT_MAX, BIN_WIDTH)
    ax.legend(loc="upper right", frameon=True, fontsize=11, prop={"weight": "bold"})
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
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if not PER_SAMPLE_CSV.is_file():
        raise SystemExit(f"Missing {PER_SAMPLE_CSV}; run mfragnet diagnostic first.")

    cache = args.out_dir / "figure5_stage_II_testing_frag_length_density_cache.npz"
    cancer_ids = _stage_ii_cancer_ids()
    cancer_paths = [TEST_FRAG / f"{sid}.csv" for sid in cancer_ids]
    for p in cancer_paths:
        if not p.is_file():
            raise SystemExit(f"Missing fragment CSV: {p}")

    if cache.exists() and not args.rebuild:
        data = np.load(cache, allow_pickle=True)
        healthy_curves = list(data["healthy_curves"])
        cancer_curves = list(data["cancer_curves"])
        print(f"Loaded cache {cache}")
    else:
        healthy_paths = _healthy_testing_paths()
        print(f"Computing densities for {len(healthy_paths)} testing healthy samples ...")
        healthy_dens = _load_densities(healthy_paths, args.workers)
        healthy_curves = [_curve(d) for d in healthy_dens]
        print(f"Computing densities for {len(cancer_paths)} Stage II cancer samples ...")
        cancer_dens = _load_densities(cancer_paths, args.workers)
        cancer_curves = [_curve(d) for d in cancer_dens]
        np.savez(
            cache,
            healthy_curves=np.array(healthy_curves, dtype=object),
            cancer_curves=np.array(cancer_curves, dtype=object),
        )
        print(f"Wrote cache {cache}")

    healthy_median = np.median(np.stack(healthy_curves, axis=0), axis=0)
    s = healthy_median.sum()
    if s > 0:
        healthy_median = healthy_median / s

    stem = "figure5_stage_II_frag_length_density_100_220"
    plot_density(
        healthy_median,
        cancer_curves,
        out_png=args.out_dir / f"{stem}.png",
        out_pdf=args.out_dir / f"{stem}.pdf",
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
