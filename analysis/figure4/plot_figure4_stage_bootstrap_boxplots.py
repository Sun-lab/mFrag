#!/usr/bin/env python3
"""
Figure 4 — stage-stratified bootstrap boxplots (Stage I / II / III+IV).

Two panel families (frag length, end 4-mer motif), each with 4 models and
DELFI vs others one-sided Mann–Whitney U (Holm-adjusted) brackets.

Bootstrap pool per stage: all healthy + lung cancer of that stage.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
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

from figure4_plot_titles import bootstrap_plot_title

CFCRAFT = Path("/fh/working/sun_w/saishi_cui/cfCraft")
MATHIOS = CFCRAFT / "Mathios"
SL_DIR = MATHIOS / "Mathios_slratio_method"
FD_DIR = MATHIOS / "FDConv"
META_PATH = MATHIOS / "metadata" / "validation_testing_sample_list.tsv"
OUT_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure4"

REF_KEY = "delfi"
BRACKET_Y_2 = {"auc": (1.02, 1.18), "prauc": (0.88, 1.06)}
BRACKET_Y_3 = {"auc": (1.02, 1.10, 1.18), "prauc": (0.88, 0.98, 1.08)}
FONT_BASE = 14
FONT_TITLE = 17
FONT_LABEL = 15
FONT_TICK = 13
FONT_LEGEND = 12
FONT_BRACKET = 12
BOX_SPAN = 0.46
BOX_WIDTH = 0.14
BOX_XPAD = 0.20
STAGE_GROUPS = (
    ("stage_I", "stage_I", "Stage I"),
    ("stage_II", "stage_II", "Stage II"),
    ("stage_III_IV", "stage_III_IV", "Stage III + IV"),
)


@dataclass(frozen=True)
class PanelConfig:
    slug: str
    title_prefix: str
    groups: tuple[tuple[str, str], ...]
    colors: dict[str, str]
    models: tuple[tuple[str, str, Path | None, Path | None, int], ...]


FRAG_PANEL = PanelConfig(
    slug="frag_length",
    title_prefix="Frag length",
    groups=(
        ("delfi", "DELFI (S/L ratio)"),
        ("mfrag_delfi", "mFrag-Weight (S/L ratio)"),
        ("mfrag_net_ref", "mFrag-Net (Frag length)"),
    ),
    colors={
        "delfi": "#1f77b4",
        "mfrag_delfi": "#ff7f0e",
        "mfrag_net_ref": "#9467bd",
    },
    models=(
        ("delfi", "lasso", SL_DIR / "pca66_l1_baseline" / "test_predictions.csv", None, 1),
        (
            "mfrag_delfi",
            "lasso",
            SL_DIR / "pca66_l1_coupling_abs_dbeta_over_mean" / "test_predictions.csv",
            None,
            2,
        ),
        ("mfrag_net_ref", "cnn", None, FD_DIR / "frag_meth3ch_new" / "runs" / "rep_031.npz", 31),
    ),
)

END4MOTIF_PANEL = PanelConfig(
    slug="end4motif",
    title_prefix="End 4-mer motif",
    groups=(
        ("delfi", "DELFI (MDS)"),
        ("mfrag_delfi", "mFrag-Weight (MDS)"),
        ("mfrag_net_ref", "mFrag-Net (Motif)"),
    ),
    colors={
        "delfi": "#1f77b4",
        "mfrag_delfi": "#ff7f0e",
        "mfrag_net_ref": "#9467bd",
    },
    models=(
        ("delfi", "lasso", SL_DIR / "shannon_pca54_l1_baseline" / "test_predictions.csv", None, 1),
        (
            "mfrag_delfi",
            "lasso",
            SL_DIR / "shannon_pca54_l1_coupling_one_minus_lambda" / "test_predictions.csv",
            None,
            2,
        ),
        ("mfrag_net_ref", "cnn", None, FD_DIR / "motif_meth3ch_new" / "runs" / "rep_006.npz", 6),
    ),
)

PANELS = (FRAG_PANEL, END4MOTIF_PANEL)
GROUP_LABELS = {g[0]: g[1] for panel in PANELS for g in panel.groups}


def _comp_keys(panel: PanelConfig) -> tuple[str, ...]:
    return tuple(k for k, _ in panel.groups if k != REF_KEY)


def _bracket_y(panel: PanelConfig) -> dict[str, tuple[float, ...]]:
    return BRACKET_Y_2 if len(_comp_keys(panel)) == 2 else BRACKET_Y_3


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


def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64)
    m = int(p.size)
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(m, dtype=np.float64)
    prev = 0.0
    for i, pv in enumerate(ranked):
        k = m - i
        val = min(1.0, pv * k)
        val = max(val, prev)
        adj[i] = val
        prev = val
    out = np.empty(m, dtype=np.float64)
    out[order] = adj
    return out


def _stage_group(stage: str) -> str | None:
    s = str(stage).strip().upper()
    if s == "I":
        return "stage_I"
    if s == "II":
        return "stage_II"
    if s in ("III", "IV"):
        return "stage_III_IV"
    return None


def _load_lasso(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    y = df["y_true"].to_numpy(dtype=np.int32)
    p = df["predicted_proba_class1"].to_numpy(dtype=np.float64)
    names = df["sample_id"].astype(str).tolist()
    return y, p, names


def _load_cnn(npz_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    z = np.load(npz_path, allow_pickle=True)
    y = np.asarray(z["labels"], dtype=np.int32)
    p = np.asarray(z["probs"], dtype=np.float64)
    names = [str(f).replace(".csv", "") for f in z["sample_files"]]
    return y, p, names


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


def _load_panel_models(panel: PanelConfig) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ref_names: list[str] | None = None
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for mkey, kind, csv_path, npz_path, _ in panel.models:
        if kind == "lasso":
            y, p, names = _load_lasso(csv_path)
        else:
            y, p, names = _load_cnn(npz_path)
        if ref_names is None:
            ref_names = names
        elif names != ref_names:
            raise ValueError(f"{panel.slug}: sample_id order mismatch for {mkey}")
        out[mkey] = (y, p)
    return out


def _boot_seed(base_seed: int, panel_slug: str, stage_key: str, seed_tag: int) -> int:
    return base_seed + hash((panel_slug, stage_key, seed_tag)) % 10000


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


def _mwu_delfi_vs_others(
    data: dict[str, dict[str, np.ndarray]],
    metric: str,
    comp_keys: tuple[str, ...],
) -> pd.DataFrame:
    ref = data[REF_KEY][metric]
    rows = []
    for comp in comp_keys:
        other = data[comp][metric]
        res = mannwhitneyu(other, ref, alternative="greater", method="auto")
        rows.append(
            {
                "metric": metric,
                "comparator_key": comp,
                "comparator_label": GROUP_LABELS[comp].replace("\n", " "),
                "u_stat": float(res.statistic),
                "p_raw": float(res.pvalue),
                "median_delfi": float(np.median(ref)),
                "median_comparator": float(np.median(other)),
                "alternative": "comparator > DELFI",
            }
        )
    df = pd.DataFrame(rows)
    df["p_holm"] = _holm_adjust(df["p_raw"].to_numpy())
    df["sig_holm"] = df["p_holm"].map(_sig_star)
    return df


def _plot_boxplot(
    panel: PanelConfig,
    data: dict[str, dict[str, np.ndarray]],
    *,
    out_png: Path,
    out_pdf: Path,
    ylim: tuple[float, float],
    dpi: int,
    mwu_auc: pd.DataFrame,
    mwu_pr: pd.DataFrame,
    plot_title: str,
    bracket_y: dict[str, tuple[float, ...]] | None = None,
) -> None:
    metric_keys = ("auc", "prauc")
    metric_labels = ("Testing AUC", "Testing PRAUC")
    group_centers = np.array([1.0, 2.0], dtype=float)
    n_boxes = len(panel.groups)
    span = BOX_SPAN if n_boxes == 3 else 0.62
    offsets = np.linspace(-span / 2, span / 2, n_boxes)
    width = BOX_WIDTH

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
        for gi, (gkey, _) in enumerate(panel.groups):
            pos = gc + offsets[gi]
            positions.append(pos)
            box_data.append(data[gkey][mkey])
            colors_order.append(gkey)
            pos_map[(mkey, gkey)] = pos

    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 2.2},
        whiskerprops={"color": "black", "linewidth": 1.5},
        capprops={"color": "black", "linewidth": 1.5},
        boxprops={"edgecolor": "black", "linewidth": 1.5},
    )
    for i, gkey in enumerate(colors_order):
        bp["boxes"][i].set_facecolor(panel.colors[gkey])
        bp["boxes"][i].set_alpha(0.35)

    rng = np.random.default_rng(7)
    max_pts = 60
    y_lo, y_hi = ylim
    for mkey in metric_keys:
        for gkey, _ in panel.groups:
            pos = pos_map[(mkey, gkey)]
            pts = np.asarray(data[gkey][mkey], dtype=np.float64)
            pts = pts[(pts >= y_lo) & (pts <= y_hi) & np.isfinite(pts)]
            if pts.size == 0:
                continue
            k = min(pts.size, max_pts)
            if k < pts.size:
                pts = pts[rng.choice(pts.size, size=k, replace=False)]
            base = panel.colors[gkey]
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

    bracket_y = bracket_y or BRACKET_Y_3

    for mkey, mwu_df in (("auc", mwu_auc), ("prauc", mwu_pr)):
        levels = bracket_y[mkey]
        x_ref = pos_map[(mkey, REF_KEY)]
        for bi, row in mwu_df.reset_index(drop=True).iterrows():
            x_comp = pos_map[(mkey, row["comparator_key"])]
            x1, x2 = sorted((x_comp, x_ref))
            y_line = levels[bi] if bi < len(levels) else levels[-1]
            x_mid = (x1 + x2) / 2.0
            sig = str(row["sig_holm"])
            p_fmt = _format_p(float(row["p_holm"]))
            ax.plot([x1, x1, x2, x2], [y_line, y_line + h, y_line + h, y_line], color="black", lw=1.4)
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

    left = group_centers[0] + offsets[0] - width / 2 - BOX_XPAD
    right = group_centers[1] + offsets[-1] + width / 2 + BOX_XPAD
    ax.set_xlim(left, right)

    legend_handles = [
        Patch(
            facecolor=panel.colors[k],
            edgecolor="black",
            linewidth=1.2,
            alpha=0.35,
            label=lbl.replace("\n", " "),
        )
        for k, lbl in panel.groups
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.95, edgecolor="0.75")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Wrote {out_png}")


def _run_panel(panel: PanelConfig, *, out_dir: Path, n_boot: int, seed: int, dpi: int, ylim: tuple[float, float]) -> None:
    _, _, sample_ids = _load_lasso(panel.models[0][2])
    pools, pool_summary = _build_stage_pools(sample_ids)
    pool_summary.to_csv(out_dir / f"{panel.slug}_stage_pool_sizes.csv", index=False)
    model_data = _load_panel_models(panel)

    all_mwu = []
    all_summary = []
    all_draws = []

    comp_keys = _comp_keys(panel)
    bracket_default = _bracket_y(panel)

    for stage_key, stage_slug, stage_label in STAGE_GROUPS:
        pool_idx = pools[stage_key]
        row = pool_summary.loc[pool_summary["stage_group"] == stage_key].iloc[0]
        n_pool = int(row["n_pool"])
        n_cancer = int(row["n_cancer_in_stage"])
        n_healthy = int(row["n_healthy"])

        data: dict[str, dict[str, np.ndarray]] = {}
        for mkey, _, csv_path, npz_path, seed_tag in panel.models:
            y, prob = model_data[mkey]
            seed_i = _boot_seed(seed, panel.slug, stage_key, seed_tag)
            auc_boot, pr_boot = _bootstrap_stage_metrics(y, prob, pool_idx, n_boot, seed_i)
            data[mkey] = {"auc": auc_boot, "prauc": pr_boot}
            print(
                f"[{panel.slug}] {stage_label} {mkey}: n={n_pool} "
                f"AUC med={np.median(auc_boot):.4f} PRAUC med={np.median(pr_boot):.4f}",
                flush=True,
            )

        mwu_auc = _mwu_delfi_vs_others(data, "auc", comp_keys)
        mwu_pr = _mwu_delfi_vs_others(data, "prauc", comp_keys)
        mwu = pd.concat([mwu_auc, mwu_pr], ignore_index=True)
        mwu.insert(0, "panel", panel.slug)
        mwu.insert(1, "stage_group", stage_key)
        mwu.insert(2, "stage_label", stage_label)
        all_mwu.append(mwu)

        for gkey, glabel in panel.groups:
            for mkey in ("auc", "prauc"):
                arr = data[gkey][mkey]
                all_summary.append(
                    {
                        "panel": panel.slug,
                        "stage_group": stage_key,
                        "stage_label": stage_label,
                        "group_key": gkey,
                        "group_label": glabel.replace("\n", " "),
                        "metric": mkey,
                        "n_pool": n_pool,
                        "n_bootstrap": len(arr),
                        "median": float(np.median(arr)),
                        "q25": float(np.percentile(arr, 25)),
                        "q75": float(np.percentile(arr, 75)),
                    }
                )
            for i, (a, p) in enumerate(zip(data[gkey]["auc"], data[gkey]["prauc"])):
                all_draws.append(
                    {
                        "panel": panel.slug,
                        "stage_group": stage_key,
                        "group_key": gkey,
                        "bootstrap_idx": i,
                        "test_auc": float(a),
                        "test_prauc": float(p),
                    }
                )

        title = bootstrap_plot_title(stage_key, stage_label, sample_ids, pool_summary)
        stem = f"figure4_{panel.slug}_bootstrap_boxplot_{stage_slug}"
        stage_ylim = ylim
        bracket_y = bracket_default
        if stage_key == "stage_III_IV" and panel.slug == "end4motif":
            stage_ylim = (0.0, 1.6)
            bracket_y = {
                "auc": (1.12, 1.28),
                "prauc": tuple(y + 0.1 for y in BRACKET_Y_2["prauc"]),
            }
        _plot_boxplot(
            panel,
            data,
            out_png=out_dir / f"{stem}.png",
            out_pdf=out_dir / f"{stem}.pdf",
            ylim=stage_ylim,
            dpi=dpi,
            mwu_auc=mwu_auc,
            mwu_pr=mwu_pr,
            plot_title=title,
            bracket_y=bracket_y,
        )

    pd.concat(all_mwu, ignore_index=True).to_csv(
        out_dir / f"figure4_{panel.slug}_stage_mwu_delfi_vs_others.csv", index=False
    )
    pd.DataFrame(all_summary).to_csv(out_dir / f"figure4_{panel.slug}_stage_bootstrap_summary.csv", index=False)
    pd.DataFrame(all_draws).to_csv(out_dir / f"figure4_{panel.slug}_stage_bootstrap_draws.csv", index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--panel", choices=("frag_length", "end4motif", "all"), default="all")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--ylim-max", type=float, default=1.55)
    args = ap.parse_args()

    ylim = (0.0, float(args.ylim_max))
    panels = PANELS if args.panel == "all" else tuple(p for p in PANELS if p.slug == args.panel)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for panel in panels:
        _run_panel(panel, out_dir=args.out_dir, n_boot=args.n_bootstrap, seed=args.seed, dpi=args.dpi, ylim=ylim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
