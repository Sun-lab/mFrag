#!/usr/bin/env python3
"""
Figure S4 — mFrag-DELFI (Frag length) vs three permuted-coupling null draws (Lasso PCA).

Uses Lasso rows 2–4 from pca66_permuted_coupling_bootstrap_compare (mean, permute_1,
permute_2), trains permute_3 if missing, then plots Testing AUC / PRAUC boxplots with
one-sided Mann–Whitney U (mFrag-DELFI > each permuted draw).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score

CFCRAFT = Path("/fh/working/sun_w/saishi_cui/cfCraft")
MATHIOS = CFCRAFT / "Mathios"
SL_DIR = MATHIOS / "Mathios_slratio_method"
BOOT_DIR = SL_DIR / "pca66_permuted_coupling_bootstrap_compare"
BOOT_DRAWS = BOOT_DIR / "bootstrap_draws.csv"
MEAN_DIR = SL_DIR / "pca66_l1_coupling_abs_dbeta_over_mean"
PERMUTE3_DIR = SL_DIR / "pca66_l1_coupling_abs_dbeta_over_mean_permute_3"
PERMUTE3_SEED = 303
OUT_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure S4"
SCRIPT_DIR = MATHIOS / "scripts"

GROUPS = (
    ("mfrag_delfi_frag_length", "mFrag-Weight (S/L ratio)"),
    ("permute_1", "mFrag-Weight (S/L ratio random permutated weights 1)"),
    ("permute_2", "mFrag-Weight (S/L ratio random permutated weights 2)"),
    ("permute_3", "mFrag-Weight (S/L ratio random permutated weights 3)"),
)
GROUP_LABELS = {k: v for k, v in GROUPS}
SOURCE_VARIANT = {
    "mfrag_delfi_frag_length": "mean",
    "permute_1": "permute_1",
    "permute_2": "permute_2",
    "permute_3": "permute_3",
}
COLORS = {
    "mfrag_delfi_frag_length": "#ff7f0e",
    "permute_1": "#2ca02c",
    "permute_2": "#9467bd",
    "permute_3": "#1f77b4",
}
REF_KEY = "mfrag_delfi_frag_length"
PERMUTE_KEYS = ("permute_1", "permute_2", "permute_3")

FONT_BASE = 14
FONT_TITLE = 17
FONT_LABEL = 15
FONT_TICK = 13
FONT_LEGEND = 9
FONT_BRACKET = 11
BOX_SPAN = 0.52
BOX_WIDTH = 0.11
BOX_XPAD = 0.28
BRACKET_Y = {"auc": (1.36, 1.22, 1.06), "prauc": (1.09, 0.95, 0.79)}
# Tighter star–p spacing on 2nd bracket; 3rd uses a wider minimum gap.
BRACKET_LINE_GAP_MUL = {"auc": (1.0, 0.55, 0.62), "prauc": (1.0, 0.55, 0.62)}
MIN_STAR_P_GAP = {"auc": (0.0, 0.0, 0.048), "prauc": (0.0, 0.0, 0.048)}
BRACKET_TEXT_HEIGHT = 0.040
BRACKET_ABOVE_MARGIN = 0.014
BRACKET_BELOW_MARGIN = 0.016


def _bracket_label_ys(
    *,
    y_line: float,
    h: float,
    text_pad: float,
    line_gap: float,
    metric_key: str,
    bracket_idx: int,
    upper_bracket_y: float | None,
) -> tuple[float, float]:
    """Place stars/p above own bracket without overlapping bracket above."""
    gap = max(
        line_gap * BRACKET_LINE_GAP_MUL[metric_key][bracket_idx],
        MIN_STAR_P_GAP[metric_key][bracket_idx],
    )
    star_y = y_line + h + text_pad
    min_star_y = y_line + h + BRACKET_BELOW_MARGIN
    star_y = max(star_y, min_star_y)
    p_y = star_y + gap

    if upper_bracket_y is not None:
        max_p_top = upper_bracket_y - BRACKET_ABOVE_MARGIN
        max_p_y = max_p_top - BRACKET_TEXT_HEIGHT
        if p_y > max_p_y:
            p_y = max_p_y
            star_y = p_y - gap
            star_y = max(star_y, min_star_y)
            if star_y + gap > max_p_y:
                p_y = star_y + min(gap, max(0.0, max_p_y - star_y))

    return star_y, p_y


def _load_bootstrap_module():
    spec = importlib.util.spec_from_file_location(
        "perm_bootstrap",
        SCRIPT_DIR / "slratio_pca66_permuted_coupling_bootstrap.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["perm_bootstrap"] = mod
    spec.loader.exec_module(mod)
    return mod


def _sig_star(p: float) -> str:
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def _format_p(p: float) -> str:
    if p < 0.001:
        return f"p={p:.1e}"
    s = f"{p:.4f}".rstrip("0").rstrip(".")
    return f"p={s}"


def _train_permute3(train_csv: Path, test_csv: Path, *, force: bool) -> None:
    if (PERMUTE3_DIR / "test_predictions.csv").is_file() and not force:
        print(f"Skip existing {PERMUTE3_DIR.name}")
        return

    mod = _load_bootstrap_module()
    X_tr, y_tr, train_names, win_cols = mod._load_xy(train_csv)
    X_te, y_te, test_names, win_cols_te = mod._load_xy(test_csv)
    if win_cols_te != win_cols:
        raise SystemExit("Train/test window column mismatch")
    w, mean_abs = mod._mean_coupling_weights(win_cols)
    w_perm, perm_idx = mod._permute_weights(w, PERMUTE3_SEED)
    meta_base = json.loads((MEAN_DIR / "best.json").read_text(encoding="utf-8"))
    meta = {
        "model": "coupling_abs_dbeta_over_mean_permute_3",
        "description": "z × permuted (|Δβ_ext|/mean) coupling weights",
        "permute_seed": PERMUTE3_SEED,
        "permute_draw": 3,
        "mean_abs_delta_beta": mean_abs,
        "n_features": len(win_cols),
    }
    PERMUTE3_DIR.mkdir(parents=True, exist_ok=True)
    (PERMUTE3_DIR / "coupling_weights.json").write_text(
        json.dumps(
            {
                "original_weight": w.tolist(),
                "permuted_weight": w_perm.tolist(),
                "permute_index": perm_idx.tolist(),
                "window_id": win_cols,
                "permute_seed": PERMUTE3_SEED,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _, payload = mod._refit_linear(
        "l1",
        X_tr * w_perm,
        y_tr,
        X_te * w_perm,
        y_te,
        train_names,
        test_names,
        PERMUTE3_DIR,
        meta,
        n_pcs=int(meta_base["n_pcs"]),
        best_c=float(meta_base["best_C"]),
    )
    print(
        f"Lasso permute_3: test AUC={payload['test_roc_auc']:.4f} "
        f"PRAUC={payload['test_prauc']:.4f}"
    )


def _bootstrap_permute3(n_bootstrap: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    mod = _load_bootstrap_module()
    y, prob = mod._load_test_pred(PERMUTE3_DIR)
    seed_i = seed + 303
    return mod._bootstrap_metrics(y, prob, n_bootstrap, seed_i)


def _load_existing_lasso_draws() -> dict[str, dict[str, np.ndarray]]:
    df = pd.read_csv(BOOT_DRAWS)
    lasso = df[df["family"] == "Lasso"]
    out: dict[str, dict[str, np.ndarray]] = {}
    for gkey, variant in SOURCE_VARIANT.items():
        if variant == "permute_3":
            continue
        sub = lasso[lasso["variant"] == variant]
        if sub.empty:
            raise SystemExit(f"Missing Lasso variant={variant} in {BOOT_DRAWS}")
        out[gkey] = {
            "auc": sub["test_auc"].to_numpy(dtype=np.float64),
            "prauc": sub["test_prauc"].to_numpy(dtype=np.float64),
        }
    return out


def _load_series(n_bootstrap: int, seed: int) -> dict[str, dict[str, np.ndarray]]:
    data = _load_existing_lasso_draws()
    auc_boot, pr_boot = _bootstrap_permute3(n_bootstrap, seed)
    data["permute_3"] = {"auc": auc_boot, "prauc": pr_boot}
    return data


def _mwu_ref_vs_permutes(data: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for metric in ("auc", "prauc"):
        ref = data[REF_KEY][metric]
        for pkey in PERMUTE_KEYS:
            comp = data[pkey][metric]
            res = mannwhitneyu(ref, comp, alternative="greater", method="auto")
            rows.append(
                {
                    "metric": metric,
                    "reference_key": REF_KEY,
                    "reference_label": GROUP_LABELS[REF_KEY],
                    "comparator_key": pkey,
                    "comparator_label": GROUP_LABELS[pkey],
                    "u_stat": float(res.statistic),
                    "p_value": float(res.pvalue),
                    "median_reference": float(np.median(ref)),
                    "median_comparator": float(np.median(comp)),
                    "alternative": "mFrag-Weight (S/L ratio) > permuted draw",
                    "sig": _sig_star(float(res.pvalue)),
                }
            )
    return pd.DataFrame(rows)


def _plot(
    *,
    data: dict[str, dict[str, np.ndarray]],
    out_png: Path,
    out_pdf: Path,
    ylim: tuple[float, float],
    dpi: int,
    mwu: pd.DataFrame,
    plot_title: str,
) -> None:
    metric_keys = ("auc", "prauc")
    metric_labels = ("Testing AUC", "Testing PRAUC")
    group_centers = np.array([1.0, 2.0], dtype=float)
    n_boxes = len(GROUPS)
    offsets = np.linspace(-BOX_SPAN / 2, BOX_SPAN / 2, n_boxes)

    mpl.rcParams.update(
        {
            "font.size": FONT_BASE,
            "axes.titlesize": FONT_TITLE,
            "axes.labelsize": FONT_LABEL,
            "legend.fontsize": FONT_LEGEND,
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.linewidth": 1.2,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    positions: list[float] = []
    box_data: list[np.ndarray] = []
    colors_order: list[str] = []
    pos_map: dict[tuple[str, str], float] = {}

    for mi, mkey in enumerate(metric_keys):
        gc = group_centers[mi]
        for gi, (gkey, _) in enumerate(GROUPS):
            pos = gc + offsets[gi]
            positions.append(pos)
            box_data.append(data[gkey][mkey])
            colors_order.append(gkey)
            pos_map[(mkey, gkey)] = pos

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=BOX_WIDTH,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 2.2},
        whiskerprops={"color": "black", "linewidth": 1.5},
        capprops={"color": "black", "linewidth": 1.5},
        boxprops={"edgecolor": "black", "linewidth": 1.5},
    )
    for i, gkey in enumerate(colors_order):
        bp["boxes"][i].set_facecolor(COLORS[gkey])
        bp["boxes"][i].set_alpha(0.35)

    rng = np.random.default_rng(7)
    max_pts = 60
    y_lo, y_hi = ylim
    for mkey in metric_keys:
        for gkey, _ in GROUPS:
            pos = pos_map[(mkey, gkey)]
            pts = np.asarray(data[gkey][mkey], dtype=np.float64)
            pts = pts[(pts >= y_lo) & (pts <= y_hi) & np.isfinite(pts)]
            if pts.size == 0:
                continue
            k = min(pts.size, max_pts)
            if k < pts.size:
                pts = pts[rng.choice(pts.size, size=k, replace=False)]
            base = COLORS[gkey]
            r, g, b = to_rgb(base) if mcolors.is_color_like(base) else (0.4, 0.4, 0.4)
            light = (0.55 + 0.45 * r, 0.55 + 0.45 * g, 0.55 + 0.45 * b)
            jitter = rng.uniform(-0.04, 0.04, size=pts.size)
            ax.scatter(
                np.full_like(pts, pos) + jitter,
                pts,
                s=14,
                color=light,
                alpha=0.55,
                linewidths=0.0,
                zorder=3,
            )

    y_span = ylim[1] - ylim[0]
    h = 0.018 * y_span
    text_pad = 0.010 * y_span
    line_gap = 0.038 * y_span
    for _, row in mwu.iterrows():
        mkey = str(row["metric"])
        pkey = str(row["comparator_key"])
        x1 = pos_map[(mkey, REF_KEY)]
        x2 = pos_map[(mkey, pkey)]
        x_left, x_right = sorted((x1, x2))
        bi = PERMUTE_KEYS.index(pkey)  # type: ignore[arg-type]
        y_line = BRACKET_Y[mkey][bi]
        x_mid = (x_left + x_right) / 2.0
        sig = str(row["sig"])
        p_fmt = _format_p(float(row["p_value"]))
        ax.plot(
            [x_left, x_left, x_right, x_right],
            [y_line, y_line + h, y_line + h, y_line],
            color="black",
            lw=1.4,
        )
        upper_y = BRACKET_Y[mkey][bi - 1] if bi > 0 else None
        star_y, p_y = _bracket_label_ys(
            y_line=y_line,
            h=h,
            text_pad=text_pad,
            line_gap=line_gap,
            metric_key=mkey,
            bracket_idx=bi,
            upper_bracket_y=upper_y,
        )
        ax.text(
            x_mid,
            p_y,
            rf"$\mathit{{{p_fmt}}}$",
            ha="center",
            va="bottom",
            fontsize=FONT_BRACKET,
        )
        ax.text(
            x_mid,
            star_y,
            rf"$\mathbf{{{sig}}}$",
            ha="center",
            va="bottom",
            fontsize=FONT_BRACKET,
        )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(metric_labels, fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title(plot_title, fontsize=FONT_TITLE, fontweight="bold")
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="both", which="both", width=1.2, labelsize=FONT_TICK)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

    left = group_centers[0] + offsets[0] - BOX_WIDTH / 2 - BOX_XPAD
    right = group_centers[1] + offsets[-1] + BOX_WIDTH / 2 + BOX_XPAD
    ax.set_xlim(left, right)

    legend_handles = [
        Patch(facecolor=COLORS[k], edgecolor="black", linewidth=1.2, alpha=0.35, label=lbl)
        for k, lbl in GROUPS
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.95, edgecolor="0.75")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--train-csv", type=Path, default=SL_DIR / "slratio_training_row_zscore.csv")
    ap.add_argument("--test-csv", type=Path, default=SL_DIR / "slratio_testing_row_zscore.csv")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-retrain", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    if not args.skip_train:
        _train_permute3(args.train_csv, args.test_csv, force=args.force_retrain)

    data = _load_series(args.n_bootstrap, args.seed)
    mwu = _mwu_ref_vs_permutes(data)

    summary_rows = []
    long_rows = []
    for gkey, glabel in GROUPS:
        for mkey in ("auc", "prauc"):
            arr = data[gkey][mkey]
            summary_rows.append(
                {
                    "group_key": gkey,
                    "group_label": glabel,
                    "source_variant": SOURCE_VARIANT[gkey],
                    "metric": mkey,
                    "n": len(arr),
                    "median": float(np.median(arr)),
                    "q25": float(np.percentile(arr, 25)),
                    "q75": float(np.percentile(arr, 75)),
                }
            )
            for i, v in enumerate(arr):
                long_rows.append(
                    {
                        "group_key": gkey,
                        "group_label": glabel,
                        "source_variant": SOURCE_VARIANT[gkey],
                        "metric": mkey,
                        "bootstrap_idx": i,
                        "value": float(v),
                    }
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(
        args.out_dir / "figure_s4_mfragdelfi_lasso_permuted_coupling_bootstrap_summary.csv",
        index=False,
    )
    pd.DataFrame(long_rows).to_csv(
        args.out_dir / "figure_s4_mfragdelfi_lasso_permuted_coupling_bootstrap_draws.csv",
        index=False,
    )
    mwu.to_csv(
        args.out_dir / "figure_s4_mfragdelfi_lasso_permuted_coupling_bootstrap_mwu.csv",
        index=False,
    )
    print(mwu.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    plot_title = "All stages (testing healthy = 385, testing cancer = 46)"
    _plot(
        data=data,
        out_png=args.out_dir / "figure_s4_mfragdelfi_lasso_permuted_coupling_bootstrap_boxplot.png",
        out_pdf=args.out_dir / "figure_s4_mfragdelfi_lasso_permuted_coupling_bootstrap_boxplot.pdf",
        ylim=(0.0, 1.7),
        dpi=args.dpi,
        mwu=mwu,
        plot_title=plot_title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
