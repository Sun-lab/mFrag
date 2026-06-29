#!/usr/bin/env python3
"""Figure 3: 2×2 panels — MDS boxplots (row 1) and external-ref β barplots (row 2)."""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

_frag = Path(__file__).resolve().parents[1] / "Fragment_length"
if str(_frag) not in sys.path:
    sys.path.insert(0, str(_frag))

from plot_figure2_sl_beta_boxplot_panels import (  # noqa: E402
    EXT_METH_CSV,
    FONT_LABEL,
    FONT_TICK,
    FONT_TITLE,
    H_BETA_DIR,
    L_BETA_DIR,
    _draw_boxplot_panel,
    _draw_external_beta_bar_panel,
    _load_beta_by_region,
)
from plot_shannon_entropy_by_window import (  # noqa: E402
    MIN_WINDOWS_FOR_Z,
    MOTIF_COLS,
    TRAIN_FRAG,
    WINDOW_IDS,
    _zscore_within_sample,
)

MATHIOS = Path(__file__).resolve().parents[1]
CFCRAFT = MATHIOS.parent
FIGURE3_DIR = CFCRAFT / "mFrag_Figures" / "Figure3"
HERE = Path(__file__).resolve().parent

SHANNON_CACHE = HERE / "end4motif_training_window_shannon.npz"
MDS_GENOME_CACHE = HERE / "figure3_panel_mds_genomewide.npz"
BETA_CACHE = HERE / "figure3_panel_beta_by_region.npz"

DEFAULT_OUT_PNG = FIGURE3_DIR / "figure3_mds_beta_boxplot_panels.png"
DEFAULT_OUT_PDF = FIGURE3_DIR / "figure3_mds_beta_boxplot_panels.pdf"

REGIONS = (
    {"key": "genomewide", "title": "Genome-wide", "window_id": None},
    {
        "key": "chr16_0_5",
        "title": "Chr16:0–5 Mb",
        "window_id": "chr16_0_5000000",
    },
)

MOTIFS = ["".join(p) for p in product("ACGT", repeat=4)]
MOTIF_INDEX = {m: i for i, m in enumerate(MOTIFS)}
CHUNK = 2_000_000
LOG2_256 = float(np.log2(256))


def _sample_genomewide_mds(path_str: str) -> tuple[str, str, float]:
    path = Path(path_str)
    acc = np.zeros(256, dtype=np.float64)

    for chunk in pd.read_csv(
        path,
        usecols=["weight", *MOTIF_COLS],
        chunksize=CHUNK,
    ):
        w = chunk["weight"].astype(np.float64).values
        for col in MOTIF_COLS:
            s = chunk[col].astype(str).str.upper().str.strip()
            valid = (
                chunk[col].notna()
                & s.str.len().eq(4)
                & s.str.match(r"^[ACGT]{4}$", na=False)
            )
            if not valid.any():
                continue
            wt = w[valid]
            motifs = s[valid].values
            midx = np.array([MOTIF_INDEX.get(m, -1) for m in motifs], dtype=np.int32)
            ok = midx >= 0
            if ok.any():
                np.add.at(acc, midx[ok], wt[ok])

    total = float(acc.sum())
    if total <= 0:
        mds = np.nan
    else:
        p = acc[acc > 0] / total
        mds = float(-np.sum(p * np.log2(p)) / LOG2_256)

    group = "healthy" if path.name.endswith("_healthy.csv") else "lung_cancer"
    return path.stem, group, mds


def _load_mds_genomewide(rebuild: bool, *, workers: int) -> tuple[np.ndarray, np.ndarray]:
    if MDS_GENOME_CACHE.exists() and not rebuild:
        data = np.load(MDS_GENOME_CACHE, allow_pickle=True)
        print(f"Loaded MDS cache {MDS_GENOME_CACHE}")
        return data["healthy_mds"], data["cancer_mds"]

    files = sorted(TRAIN_FRAG.glob("*_healthy.csv")) + sorted(
        TRAIN_FRAG.glob("*_lung_cancer.csv")
    )
    if not files:
        raise SystemExit(f"No training fragment CSVs under {TRAIN_FRAG}")

    healthy, cancer = [], []
    print(f"Computing genome-wide MDS from {len(files)} samples ...")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_sample_genomewide_mds, str(p)): p for p in files}
        done = 0
        for fut in as_completed(futs):
            _, group, mds = fut.result()
            (healthy if group == "healthy" else cancer).append(mds)
            done += 1
            if done % 20 == 0 or done == len(files):
                print(f"  processed {done}/{len(files)} samples", flush=True)

    h_arr = np.asarray(healthy, dtype=np.float64)
    c_arr = np.asarray(cancer, dtype=np.float64)
    np.savez(MDS_GENOME_CACHE, healthy_mds=h_arr, cancer_mds=c_arr)
    print(f"Wrote MDS cache {MDS_GENOME_CACHE}")
    return h_arr, c_arr


def _load_mds_window_zscore(window_id: str) -> tuple[np.ndarray, np.ndarray]:
    if not SHANNON_CACHE.exists():
        raise SystemExit(f"Missing {SHANNON_CACHE}; run plot_shannon_entropy_by_window.py first")

    data = np.load(SHANNON_CACHE, allow_pickle=True)
    h_raw = data["healthy_shannon"] / LOG2_256
    c_raw = data["cancer_shannon"] / LOG2_256
    all_raw = np.vstack([h_raw, c_raw])
    keep = np.all(np.isfinite(all_raw), axis=0)
    win_ids = [WINDOW_IDS[i] for i in np.where(keep)[0]]
    if len(win_ids) < MIN_WINDOWS_FOR_Z:
        raise SystemExit("Too few MDS windows after filtering")
    if window_id not in win_ids:
        raise SystemExit(f"Missing window {window_id} in Shannon cache")

    col = win_ids.index(window_id)
    h_z = _zscore_within_sample(h_raw[:, keep])
    c_z = _zscore_within_sample(c_raw[:, keep])
    h = h_z[:, col]
    c = c_z[:, col]
    return h[np.isfinite(h)], c[np.isfinite(c)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rebuild-mds", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--rebuild-beta", action="store_true")
    ap.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    ap.add_argument("--out-pdf", type=Path, default=DEFAULT_OUT_PDF)
    args = ap.parse_args()

    for path in (EXT_METH_CSV, H_BETA_DIR, L_BETA_DIR, TRAIN_FRAG):
        if not path.exists():
            raise SystemExit(f"Missing required path: {path}")

    import matplotlib.pyplot as plt

    mds_genome = _load_mds_genomewide(args.rebuild_mds, workers=args.workers)
    mds_windows = {
        r["window_id"]: _load_mds_window_zscore(r["window_id"])
        for r in REGIONS
        if r["window_id"] is not None
    }
    beta_by_region = _load_beta_by_region(
        args.rebuild_beta,
        regions=REGIONS,
        cache_path=BETA_CACHE,
    )

    plt.rcParams.update(
        {
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "font.size": FONT_TICK,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), sharex=False)
    fig.subplots_adjust(left=0.11, right=0.96, top=0.93, bottom=0.08, wspace=0.42, hspace=0.38)

    for col, region in enumerate(REGIONS):
        wid = region["window_id"]
        title = region["title"]

        if wid is None:
            h_mds, c_mds = mds_genome
            h_mds = h_mds[np.isfinite(h_mds)]
            c_mds = c_mds[np.isfinite(c_mds)]
            mds_ylim = None
            zscore_line = False
            p_sci = True
        else:
            h_mds, c_mds = mds_windows[wid]
            mds_ylim = None
            zscore_line = True
            p_sci = True

        _draw_boxplot_panel(
            axes[0, col],
            h_mds,
            c_mds,
            seed=args.seed + col,
            title=title,
            ylim=mds_ylim,
            zscore_line=zscore_line,
            p_scientific=p_sci,
        )

        h_b, c_b = beta_by_region[wid]
        _draw_external_beta_bar_panel(
            axes[1, col],
            h_b,
            c_b,
            wid,
            title=title,
        )

    axes[0, 0].set_ylabel("Motif Diversity Score", fontsize=FONT_LABEL, fontweight="bold")
    axes[0, 1].set_ylabel("Motif Diversity Score (z-scored)", fontsize=FONT_LABEL, fontweight="bold")
    axes[1, 0].set_ylabel("External DNA Methyl references", fontsize=FONT_LABEL, fontweight="bold")
    axes[1, 1].set_ylabel("External DNA Methyl references", fontsize=FONT_LABEL, fontweight="bold")

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=300, bbox_inches="tight")
    fig.savefig(args.out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {args.out_png}")
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
