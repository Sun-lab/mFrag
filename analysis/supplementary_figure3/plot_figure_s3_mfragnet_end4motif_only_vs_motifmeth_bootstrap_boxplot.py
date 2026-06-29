#!/usr/bin/env python3
"""
Figure S3 — mFrag-Net end4motif-only vs end4motif+meth bootstrap boxplots.
"""
from __future__ import annotations

import argparse
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
FD_DIR = MATHIOS / "FDConv"
META_PATH = MATHIOS / "metadata" / "validation_testing_sample_list.tsv"
FIG4_DIR = CFCRAFT / "mFrag_Figures" / "Figure4"
OUT_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure S3"

if str(FIG4_DIR) not in sys.path:
    sys.path.insert(0, str(FIG4_DIR))
from figure4_plot_titles import bootstrap_plot_title  # noqa: E402

END4MOTIF_DRAWS = FD_DIR / "end4motif_new_rep093_bootstrap_compare" / "bootstrap_draws.csv"
MOTIF_METH_DRAWS = FD_DIR / "motif_meth3ch_new_rep006_bootstrap_compare" / "bootstrap_draws.csv"
END4MOTIF_NPZ = FD_DIR / "end4motif_new" / "runs" / "rep_093.npz"
MOTIF_METH_NPZ = FD_DIR / "motif_meth3ch_new" / "runs" / "rep_006.npz"

GROUPS = (
    ("mfrag_net_end4motif_only", "mFrag-Net (End 4-mer motif without DNA Methyl)"),
    ("mfrag_net_end4motif_meth", "mFrag-Net (End 4-mer motif)"),
)
GROUP_LABELS = {k: v for k, v in GROUPS}
COLORS = {
    "mfrag_net_end4motif_only": "#2ca02c",
    "mfrag_net_end4motif_meth": "#9467bd",
}
REF_KEY = "mfrag_net_end4motif_meth"
COMP_KEY = "mfrag_net_end4motif_only"

STAGE_GROUPS = (
    ("stage_I", "stage_I", "Stage I"),
    ("stage_II", "stage_II", "Stage II"),
    ("stage_III_IV", "stage_III_IV", "Stage III + IV"),
)

FONT_BASE = 14
FONT_TITLE = 17
FONT_LABEL = 15
FONT_TICK = 13
FONT_LEGEND = 10
FONT_BRACKET = 12
BOX_SPAN = 0.34
BOX_WIDTH = 0.14
BOX_XPAD = 0.22
DEFAULT_BRACKET_Y = {"auc": 1.02, "prauc": 0.88}
DEFAULT_YLIM = (0.0, 1.2)
STAGE_PLOT_OVERRIDES: dict[str, dict] = {
    "stage_III_IV": {
        "ylim": (0.0, 1.6),
        "bracket_y": {"auc": 1.12, "prauc": 1.03},
    },
}


def _stage_plot_opts(stage_key: str | None) -> tuple[tuple[float, float], dict[str, float]]:
    if stage_key is None:
        return DEFAULT_YLIM, DEFAULT_BRACKET_Y
    override = STAGE_PLOT_OVERRIDES.get(stage_key, {})
    ylim = override.get("ylim", DEFAULT_YLIM)
    bracket_y = override.get("bracket_y", DEFAULT_BRACKET_Y)
    return ylim, bracket_y


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


def _stage_group(stage: str) -> str | None:
    s = str(stage).strip().upper()
    if s == "I":
        return "stage_I"
    if s == "II":
        return "stage_II"
    if s in ("III", "IV"):
        return "stage_III_IV"
    return None


def _load_cnn(npz_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    z = np.load(npz_path, allow_pickle=True)
    y = np.asarray(z["labels"], dtype=np.int32)
    p = np.asarray(z["probs"], dtype=np.float64)
    names = [str(f).replace(".csv", "") for f in z["sample_files"]]
    return y, p, names


def _load_series_from_draws() -> dict[str, dict[str, np.ndarray]]:
    end4 = pd.read_csv(END4MOTIF_DRAWS)
    motif = pd.read_csv(MOTIF_METH_DRAWS)
    return {
        COMP_KEY: {
            "auc": end4["test_auc"].to_numpy(dtype=np.float64),
            "prauc": end4["test_prauc"].to_numpy(dtype=np.float64),
        },
        REF_KEY: {
            "auc": motif["test_auc"].to_numpy(dtype=np.float64),
            "prauc": motif["test_prauc"].to_numpy(dtype=np.float64),
        },
    }


def _build_stage_pools(sample_ids: list[str]) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    meta = pd.read_csv(META_PATH, sep="\t")
    meta["cg_id"] = meta["sample_id"].astype(str)
    cg_ids = [s.replace("_healthy", "").replace("_lung_cancer", "") for s in sample_ids]
    y = np.asarray([0 if s.endswith("_healthy") else 1 for s in sample_ids], dtype=np.int32)
    stage_map = meta.set_index("cg_id")["Stage"].astype(str).to_dict()
    stages = [stage_map.get(c, "") for c in cg_ids]
    stage_grp = [_stage_group(st) for st in stages]
    healthy_idx = np.flatnonzero(y == 0)
    pools: dict[str, np.ndarray] = {}
    rows = []
    for key, _, label in STAGE_GROUPS:
        cancer_idx = np.flatnonzero((y == 1) & np.array([g == key for g in stage_grp]))
        pool = np.sort(np.concatenate([healthy_idx, cancer_idx]))
        pools[key] = pool
        rows.append(
            {
                "stage_group": key,
                "stage_label": label,
                "n_healthy": int(healthy_idx.size),
                "n_cancer_in_stage": int(cancer_idx.size),
                "n_pool": int(pool.size),
            }
        )
    return pools, pd.DataFrame(rows)


def _boot_seed(base_seed: int, stage_key: str, model_tag: int) -> int:
    return base_seed + hash((stage_key, model_tag)) % 10000


def _bootstrap_stage_metrics(
    y: np.ndarray,
    prob: np.ndarray,
    pool_idx: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pool = np.asarray(pool_idx, dtype=np.int64)
    n = int(pool.size)
    aucs: list[float] = []
    praucs: list[float] = []
    for _ in range(n_boot):
        draw = pool[rng.integers(0, n, size=n)]
        y_b, p_b = y[draw], prob[draw]
        if len(np.unique(y_b)) < 2:
            continue
        aucs.append(float(roc_auc_score(y_b, p_b)))
        praucs.append(float(average_precision_score(y_b, p_b)))
    return np.asarray(aucs), np.asarray(praucs)


def _mwu_motifmeth_vs_end4only(data: dict[str, dict[str, np.ndarray]], metric: str) -> pd.Series:
    ref = data[REF_KEY][metric]
    comp = data[COMP_KEY][metric]
    res = mannwhitneyu(ref, comp, alternative="greater", method="auto")
    return pd.Series(
        {
            "metric": metric,
            "reference_key": REF_KEY,
            "reference_label": GROUP_LABELS[REF_KEY],
            "comparator_key": COMP_KEY,
            "comparator_label": GROUP_LABELS[COMP_KEY],
            "u_stat": float(res.statistic),
            "p_value": float(res.pvalue),
            "median_motif_meth": float(np.median(ref)),
            "median_end4motif_only": float(np.median(comp)),
            "alternative": "motif+meth > motif-only",
        }
    )


def _plot_boxplot(
    data: dict[str, dict[str, np.ndarray]],
    *,
    out_png: Path,
    out_pdf: Path,
    dpi: int,
    mwu: pd.DataFrame,
    plot_title: str,
    ylim: tuple[float, float],
    bracket_y: dict[str, float],
) -> None:
    metric_keys = ("auc", "prauc")
    metric_labels = ("Testing AUC", "Testing PRAUC")
    group_centers = np.array([1.0, 2.0], dtype=float)
    offsets = np.linspace(-BOX_SPAN / 2, BOX_SPAN / 2, len(GROUPS))

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

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
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
        x1 = pos_map[(mkey, COMP_KEY)]
        x2 = pos_map[(mkey, REF_KEY)]
        x_left, x_right = sorted((x1, x2))
        y_line = bracket_y[mkey]
        x_mid = (x_left + x_right) / 2.0
        sig = _sig_star(float(row["p_value"]))
        p_fmt = _format_p(float(row["p_value"]))
        ax.plot(
            [x_left, x_left, x_right, x_right],
            [y_line, y_line + h, y_line + h, y_line],
            color="black",
            lw=1.4,
        )
        y_base = y_line + h + text_pad
        ax.text(
            x_mid,
            y_base + line_gap,
            rf"$\mathit{{{p_fmt}}}$",
            ha="center",
            va="bottom",
            fontsize=FONT_BRACKET,
        )
        ax.text(
            x_mid,
            y_base,
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


def _record_summary_draws(
    data: dict[str, dict[str, np.ndarray]],
    *,
    stage_group: str | None,
    stage_label: str | None,
    n_pool: int | None,
) -> tuple[list[dict], list[dict]]:
    summary_rows = []
    long_rows = []
    for gkey, glabel in GROUPS:
        for mkey in ("auc", "prauc"):
            arr = data[gkey][mkey]
            summary_rows.append(
                {
                    "stage_group": stage_group or "overall",
                    "stage_label": stage_label or "All stages",
                    "group_key": gkey,
                    "group_label": glabel,
                    "metric": mkey,
                    "n_pool": n_pool,
                    "n_bootstrap": len(arr),
                    "median": float(np.median(arr)),
                    "q25": float(np.percentile(arr, 25)),
                    "q75": float(np.percentile(arr, 75)),
                }
            )
        for i, (a, p) in enumerate(zip(data[gkey]["auc"], data[gkey]["prauc"])):
            long_rows.append(
                {
                    "stage_group": stage_group or "overall",
                    "group_key": gkey,
                    "bootstrap_idx": i,
                    "test_auc": float(a),
                    "test_prauc": float(p),
                }
            )
    return summary_rows, long_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_mwu: list[pd.DataFrame] = []
    all_summary: list[dict] = []
    all_draws: list[dict] = []

    overall_data = _load_series_from_draws()
    overall_mwu = pd.DataFrame([_mwu_motifmeth_vs_end4only(overall_data, m) for m in ("auc", "prauc")])
    overall_mwu["sig"] = overall_mwu["p_value"].map(_sig_star)
    overall_mwu.insert(0, "stage_group", "overall")
    overall_mwu.insert(1, "stage_label", "All stages")
    all_mwu.append(overall_mwu)
    print(overall_mwu.to_string(index=False, float_format=lambda x: f"{x:.4g}"), flush=True)

    s_rows, d_rows = _record_summary_draws(overall_data, stage_group=None, stage_label=None, n_pool=431)
    all_summary.extend(s_rows)
    all_draws.extend(d_rows)

    sample_ids = [str(f).replace(".csv", "") for f in np.load(END4MOTIF_NPZ, allow_pickle=True)["sample_files"]]
    overall_title = bootstrap_plot_title("overall", "All stages", sample_ids)
    overall_ylim, overall_bracket = _stage_plot_opts(None)
    _plot_boxplot(
        overall_data,
        out_png=args.out_dir / "figure_s3_mfragnet_end4motif_only_vs_motifmeth_bootstrap_boxplot.png",
        out_pdf=args.out_dir / "figure_s3_mfragnet_end4motif_only_vs_motifmeth_bootstrap_boxplot.pdf",
        dpi=args.dpi,
        mwu=overall_mwu,
        plot_title=overall_title,
        ylim=overall_ylim,
        bracket_y=overall_bracket,
    )

    pools, pool_summary = _build_stage_pools(sample_ids)
    pool_summary.to_csv(args.out_dir / "figure_s3_mfragnet_end4motif_stage_pool_sizes.csv", index=False)

    y_end4, p_end4, _ = _load_cnn(END4MOTIF_NPZ)
    y_motif, p_motif, names_motif = _load_cnn(MOTIF_METH_NPZ)
    if names_motif != sample_ids:
        raise SystemExit("end4motif vs motif+meth sample_id order mismatch")
    models = {
        COMP_KEY: (y_end4, p_end4, 93),
        REF_KEY: (y_motif, p_motif, 6),
    }

    for stage_key, stage_slug, stage_label in STAGE_GROUPS:
        pool_idx = pools[stage_key]
        data: dict[str, dict[str, np.ndarray]] = {}
        for mkey, (y, prob, seed_tag) in models.items():
            seed_i = _boot_seed(args.seed, stage_key, seed_tag)
            auc_boot, pr_boot = _bootstrap_stage_metrics(y, prob, pool_idx, args.n_bootstrap, seed_i)
            data[mkey] = {"auc": auc_boot, "prauc": pr_boot}
            print(
                f"[{stage_label}] {mkey}: n={pool_idx.size} "
                f"AUC med={np.median(auc_boot):.4f} PRAUC med={np.median(pr_boot):.4f}",
                flush=True,
            )

        mwu = pd.DataFrame([_mwu_motifmeth_vs_end4only(data, m) for m in ("auc", "prauc")])
        mwu["sig"] = mwu["p_value"].map(_sig_star)
        mwu.insert(0, "stage_group", stage_key)
        mwu.insert(1, "stage_label", stage_label)
        all_mwu.append(mwu)
        print(mwu.to_string(index=False, float_format=lambda x: f"{x:.4g}"), flush=True)

        row = pool_summary.loc[pool_summary["stage_group"] == stage_key].iloc[0]
        s_rows, d_rows = _record_summary_draws(
            data,
            stage_group=stage_key,
            stage_label=stage_label,
            n_pool=int(row["n_pool"]),
        )
        all_summary.extend(s_rows)
        all_draws.extend(d_rows)

        title = bootstrap_plot_title(stage_key, stage_label, sample_ids, pool_summary)
        stage_ylim, stage_bracket = _stage_plot_opts(stage_key)
        stem = f"figure_s3_mfragnet_end4motif_only_vs_motifmeth_{stage_slug}_bootstrap_boxplot"
        _plot_boxplot(
            data,
            out_png=args.out_dir / f"{stem}.png",
            out_pdf=args.out_dir / f"{stem}.pdf",
            dpi=args.dpi,
            mwu=mwu,
            plot_title=title,
            ylim=stage_ylim,
            bracket_y=stage_bracket,
        )

    pd.concat(all_mwu, ignore_index=True).to_csv(
        args.out_dir / "figure_s3_mfragnet_end4motif_only_vs_motifmeth_bootstrap_mwu.csv",
        index=False,
    )
    pd.DataFrame(all_summary).to_csv(
        args.out_dir / "figure_s3_mfragnet_end4motif_only_vs_motifmeth_bootstrap_summary.csv",
        index=False,
    )
    pd.DataFrame(all_draws).to_csv(
        args.out_dir / "figure_s3_mfragnet_end4motif_only_vs_motifmeth_bootstrap_draws.csv",
        index=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
