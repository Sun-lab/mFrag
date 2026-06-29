#!/usr/bin/env python3
"""
Per-sample fragment-length density within two selected 5 Mb windows.

Window A: largest cohort mean S/L z-diff with lung > healthy (training cfDNA).
Window B: smallest |S/L z-diff| (lung − healthy) across cohorts.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d

_scripts = Path(__file__).resolve().parents[1] / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from nmb_window_ratios_core import CHROM_SIZES  # noqa: E402
from fragment_length_plot_annotations import annotate_short_long_split  # noqa: E402
from plot_sl_ratio_by_window import (  # noqa: E402
    MIN_WINDOWS_FOR_Z,
    WINDOW_IDS,
    _zscore_within_sample,
)

MATHIOS = Path(__file__).resolve().parents[1]
TRAIN_FRAG = MATHIOS / "fragments_csv" / "training"
OUT_DIR = Path(__file__).resolve().parent
SL_CACHE = OUT_DIR / "sl_ratio_training_window_all.npz"
SL_WINDOW_CACHE = OUT_DIR / "sl_ratio_training_window_100_220.npz"

LEN_MIN = 30
LEN_MAX = 500
N_LEN = LEN_MAX - LEN_MIN + 1
CHUNK = 2_000_000
SMOOTH_SIGMA = 1.5
PLOT_MIN = 100
PLOT_MAX = 220
SHORT_MIN, SHORT_MAX = 100, 150
LONG_MIN, LONG_MAX = 151, 220

COLOR_HEALTHY = "#2166AC"
COLOR_CANCER = "#B2182B"
FILL_HEALTHY = "#92C5DE"
FILL_CANCER = "#F4A582"
Y_MAX = 1.0
CANCER_RATIO_CAP = 1.0
BOX_LW = 2.2


def _window_label_mb(window_id: str) -> str:
    chrom, start, end = parse_window_id(window_id)
    chrom_disp = chrom[0].upper() + chrom[1:]

    def _mb(pos: int) -> str:
        v = pos / 1_000_000
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.1f}".rstrip("0").rstrip(".")

    return f"{chrom_disp}:{_mb(start)}Mb-{_mb(end)}Mb"


def _title_from_window_id(window_id: str) -> str:
    return f"{_window_label_mb(window_id)} window fragment length distribution"


def _boxplot_title_from_window_id(window_id: str, *, zscore: bool) -> str:
    label = _window_label_mb(window_id)
    if zscore:
        return f"{label} Short/Long ratio (z-scored)"
    return f"{label} Short/Long ratio"


def parse_window_id(window_id: str) -> tuple[str, int, int]:
    parts = window_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"Bad window_id: {window_id}")
    return parts[0], int(parts[1]), int(parts[2])


def _cohort_sl_diff() -> tuple[list[str], np.ndarray]:
    if not SL_CACHE.exists():
        raise SystemExit(f"Missing {SL_CACHE}; run plot_sl_ratio_by_window.py first")

    data = np.load(SL_CACHE, allow_pickle=True)
    h_raw, c_raw = data["healthy_sl"], data["cancer_sl"]
    all_raw = np.vstack([h_raw, c_raw])
    keep = np.all(np.isfinite(all_raw), axis=0)
    win_ids = [WINDOW_IDS[i] for i in np.where(keep)[0]]
    if len(win_ids) < MIN_WINDOWS_FOR_Z:
        raise SystemExit("Too few S/L windows after filtering")

    h_z = _zscore_within_sample(h_raw[:, keep])
    c_z = _zscore_within_sample(c_raw[:, keep])
    sl_diff = np.nanmean(c_z, axis=0) - np.nanmean(h_z, axis=0)
    return win_ids, sl_diff


def select_windows() -> tuple[dict, dict]:
    win_ids, sl_diff = _cohort_sl_diff()

    pos_idx = np.where(sl_diff > 0)[0]
    if pos_idx.size == 0:
        raise SystemExit("No window with lung S/L > healthy")
    max_i = int(pos_idx[np.argmax(sl_diff[pos_idx])])
    min_i = int(np.argmin(np.abs(sl_diff)))

    max_meta = {
        "window_id": win_ids[max_i],
        "title": _title_from_window_id(win_ids[max_i]),
        "tag": "max_sl_lung_gt_healthy",
    }
    min_meta = {
        "window_id": win_ids[min_i],
        "title": _title_from_window_id(win_ids[min_i]),
        "tag": "min_abs_sl_lung_minus_healthy",
    }
    return max_meta, min_meta


def _sample_window_density(args: tuple[str, str, int, int]) -> tuple[str, str, np.ndarray]:
    path_str, chrom, win_start, win_end = args
    path = Path(path_str)
    hist = np.zeros(N_LEN, dtype=np.float64)

    usecols = ["chromosome", "start", "fragment_length", "weight"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK):
        df = chunk.rename(columns={"chromosome": "c"})
        df["c"] = df["c"].astype(str)
        df["c"] = df["c"].where(df["c"].str.startswith("chr"), "chr" + df["c"])
        df = df[(df["c"] == chrom) & (df["c"].isin(CHROM_SIZES.keys()))]
        if df.empty:
            continue
        start = df["start"].to_numpy(dtype=np.int64, copy=False)
        in_win = (start >= win_start) & (start < win_end)
        if not in_win.any():
            continue
        df = df.loc[in_win]

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


def _smooth_curve(dens: np.ndarray) -> np.ndarray:
    if dens.sum() <= 0:
        return dens
    y = gaussian_filter1d(dens, sigma=SMOOTH_SIGMA, mode="nearest")
    s = y.sum()
    return y / s if s > 0 else y


def _slice_renorm(dens: np.ndarray, plot_min: int, plot_max: int) -> np.ndarray:
    sub = dens[(plot_min - LEN_MIN) : (plot_max - LEN_MIN + 1)].copy()
    total = sub.sum()
    return sub / total if total > 0 else sub


def _ratio_from_density(dens: np.ndarray) -> float:
    short_w = dens[(SHORT_MIN - LEN_MIN) : (SHORT_MAX - LEN_MIN + 1)].sum()
    long_w = dens[(LONG_MIN - LEN_MIN) : (LONG_MAX - LEN_MIN + 1)].sum()
    return float(short_w / long_w) if long_w > 0 else np.nan


def _ratios_from_densities(healthy: list[np.ndarray], cancer: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    h = np.array([_ratio_from_density(d) for d in healthy], dtype=np.float64)
    c = np.array([_ratio_from_density(d) for d in cancer], dtype=np.float64)
    return h, c


def _window_zscore_sl(window_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample z-score of 5 Mb window S/L (100–220 bp); return one window column."""
    if not SL_WINDOW_CACHE.exists():
        raise SystemExit(
            f"Missing {SL_WINDOW_CACHE}; run: "
            "python3 plot_sl_ratio_by_window.py --mode 100_220"
        )
    try:
        win_i = WINDOW_IDS.index(window_id)
    except ValueError as exc:
        raise SystemExit(f"Unknown window_id: {window_id}") from exc

    data = np.load(SL_WINDOW_CACHE, allow_pickle=True)
    h_z = _zscore_within_sample(data["healthy_sl"])[:, win_i]
    c_z = _zscore_within_sample(data["cancer_sl"])[:, win_i]
    print(f"Using z-scored 5 Mb window S/L from {SL_WINDOW_CACHE} ({window_id})")
    return h_z, c_z


def _format_p(p: float, *, scientific: bool = False) -> str:
    if scientific:
        return f"{p:.2e}"
    if p < 0.001:
        return f"{p:.4f}"
    return f"{p:.3f}".rstrip("0").rstrip(".")


def _build_densities(
    window_id: str, workers: int, rebuild: bool
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    chrom, win_start, win_end = parse_window_id(window_id)
    cache = OUT_DIR / f"fragment_length_density_{window_id}.npz"
    if cache.exists() and not rebuild:
        data = np.load(cache, allow_pickle=True)
        print(f"Loaded cache {cache}")
        return list(data["healthy"]), list(data["cancer"])

    files = sorted(TRAIN_FRAG.glob("*_healthy.csv")) + sorted(
        TRAIN_FRAG.glob("*_lung_cancer.csv")
    )
    print(f"[{window_id}] Computing weighted densities for {len(files)} samples ...")
    healthy, cancer = [], []
    task_args = (chrom, win_start, win_end)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_sample_window_density, (str(p), *task_args)): p for p in files
        }
        done = 0
        for fut in as_completed(futs):
            _, group, dens = fut.result()
            (healthy if group == "healthy" else cancer).append(dens)
            done += 1
            if done % 20 == 0 or done == len(files):
                print(f"  {window_id}: {done}/{len(files)}", flush=True)

    np.savez(cache, healthy=np.array(healthy), cancer=np.array(cancer))
    print(f"Wrote cache {cache}")
    return healthy, cancer


def _plot_window(
    healthy: list[np.ndarray],
    cancer: list[np.ndarray],
    out_png: Path,
    plot_min: int = PLOT_MIN,
    plot_max: int = PLOT_MAX,
    title: str = "Fragment length distribution",
) -> None:
    x = np.arange(plot_min, plot_max + 1, dtype=np.float64)
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    fig, ax = plt.subplots(figsize=(12, 6))
    for dens in healthy:
        ax.plot(
            x,
            _smooth_curve(_slice_renorm(dens, plot_min, plot_max)),
            color=COLOR_HEALTHY,
            alpha=0.22,
            linewidth=0.7,
            zorder=1,
        )
    for dens in cancer:
        ax.plot(
            x,
            _smooth_curve(_slice_renorm(dens, plot_min, plot_max)),
            color=COLOR_CANCER,
            alpha=0.22,
            linewidth=0.7,
            zorder=2,
        )

    ax.plot([], [], color=COLOR_HEALTHY, linewidth=2, label=f"Healthy (N={len(healthy)})")
    ax.plot([], [], color=COLOR_CANCER, linewidth=2, label=f"Lung cancer (N={len(cancer)})")

    ax.set_xlim(plot_min, plot_max)
    ax.set_xlabel("Fragment length (bp)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Weighted density", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=11)
    ax.grid(True, alpha=0.25, axis="y")
    annotate_short_long_split(ax, plot_min, plot_max)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def _plot_sl_boxplot(
    healthy: np.ndarray,
    cancer: np.ndarray,
    out_png: Path,
    seed: int,
    title: str = "Short/long ratio",
    *,
    zscore: bool = False,
) -> None:
    h = healthy[np.isfinite(healthy)]
    c = cancer[np.isfinite(cancer)]
    if not zscore:
        n_c_before = len(c)
        c = c[c <= CANCER_RATIO_CAP]
        if len(c) < n_c_before:
            print(
                f"  Excluded {n_c_before - len(c)} lung cancer sample(s) "
                f"with ratio > {CANCER_RATIO_CAP}"
            )

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    fig, ax = plt.subplots(figsize=(6, 6))
    positions = [1, 2]
    data = [h, c]
    labels = ["Healthy", "Lung cancer"]

    box_kw = dict(
        patch_artist=True,
        widths=0.5,
        showfliers=False,
        boxprops=dict(linewidth=BOX_LW, edgecolor="black"),
        whiskerprops=dict(linewidth=BOX_LW, color="black"),
        capprops=dict(linewidth=BOX_LW, color="black"),
        medianprops=dict(color="black", linewidth=BOX_LW),
    )
    try:
        bp = ax.boxplot(data, positions=positions, tick_labels=labels, **box_kw)
    except TypeError:
        bp = ax.boxplot(data, positions=positions, labels=labels, **box_kw)

    for patch, fill in zip(bp["boxes"], (FILL_HEALTHY, FILL_CANCER)):
        patch.set_facecolor(fill)
        patch.set_edgecolor("black")
        patch.set_linewidth(BOX_LW)
        patch.set_alpha(0.85)

    rng = np.random.default_rng(seed)
    for pos, vals, color in zip(positions, data, (COLOR_HEALTHY, COLOR_CANCER)):
        jitter = pos + 0.11 * rng.standard_normal(len(vals))
        ax.scatter(jitter, vals, s=18, alpha=0.45, c=color, edgecolors="none", zorder=3)

    if len(h) >= 1 and len(c) >= 1:
        _, p = stats.mannwhitneyu(h, c, alternative="two-sided")
        ax.text(
            0.5,
            0.98,
            f"Mann-Whitney U, p = {_format_p(p, scientific=zscore)}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    if zscore:
        ax.axhline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.35)
        ylabel = "Short/long ratio (z-score)"
    else:
        ax.set_ylim(0, Y_MAX)
        ylabel = "Short/long ratio"
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"  Healthy: n={len(h)}, median={np.median(h):.4f}")
    print(f"  Lung cancer: n={len(c)}, median={np.median(c):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument(
        "--selection",
        choices=("both", "max", "min"),
        default="both",
        help="max = lung>healthy S/L window; min = smallest |S/L diff|",
    )
    ap.add_argument("--plot-min", type=int, default=PLOT_MIN)
    ap.add_argument("--plot-max", type=int, default=PLOT_MAX)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--boxplot-raw",
        action="store_true",
        help="Boxplot: raw window S/L (default: within-sample z-score across 5 Mb windows, 100–220 bp)",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    max_meta, min_meta = select_windows()
    print("Selected windows:")
    print(f"  SL max (lung > healthy): {max_meta['window_id']}")
    print(f"  |SL lung − SL healthy| min: {min_meta['window_id']}")

    all_selections = [
        (
            max_meta,
            OUT_DIR / f"fragment_length_density_{max_meta['window_id']}_{max_meta['tag']}.png",
            OUT_DIR / f"fragment_length_sl_ratio_boxplot_{max_meta['window_id']}_{max_meta['tag']}.png",
            OUT_DIR / f"sl_ratio_{max_meta['window_id']}_{max_meta['tag']}_samples.csv",
        ),
        (
            min_meta,
            OUT_DIR / f"fragment_length_density_{min_meta['window_id']}_{min_meta['tag']}.png",
            OUT_DIR / f"fragment_length_sl_ratio_boxplot_{min_meta['window_id']}_{min_meta['tag']}.png",
            OUT_DIR / f"sl_ratio_{min_meta['window_id']}_{min_meta['tag']}_samples.csv",
        ),
    ]
    if args.selection == "max":
        selections = [all_selections[0]]
    elif args.selection == "min":
        selections = [all_selections[1]]
    else:
        selections = all_selections

    summary_lines = []
    for meta, out_density, out_box, out_csv in selections:
        h_dens, c_dens = _build_densities(meta["window_id"], args.workers, args.rebuild)
        _plot_window(
            h_dens, c_dens, out_density, args.plot_min, args.plot_max, title=meta["title"]
        )
        use_zscore = not args.boxplot_raw
        if use_zscore:
            h_sl, c_sl = _window_zscore_sl(meta["window_id"])
            col_name = "short_long_ratio_zscore"
        else:
            h_sl, c_sl = _ratios_from_densities(h_dens, c_dens)
            col_name = "short_long_ratio"
        box_title = _boxplot_title_from_window_id(meta["window_id"], zscore=use_zscore)
        rows = (
            [("healthy", float(v)) for v in h_sl if np.isfinite(v)]
            + [("lung_cancer", float(v)) for v in c_sl if np.isfinite(v)]
        )
        pd.DataFrame(rows, columns=["group", col_name]).to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")
        _plot_sl_boxplot(h_sl, c_sl, out_box, args.seed, title=box_title, zscore=use_zscore)
        summary_lines.append(
            f"{meta['window_id']}\t{meta['title']}\t{out_density.name}\t{out_box.name}"
        )

    summary_path = OUT_DIR / "fragment_length_density_selected_windows.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
