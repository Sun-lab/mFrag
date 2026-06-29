#!/usr/bin/env python3
"""
Figure 6 — detection depth bootstrap boxplots for mFrag-Net (Frag length) and Motif.

Fixed trained models:
  frag_meth3ch_new rep031
  motif_meth3ch_new rep006

Testing depths: 0.1%, 1%, 5%, 10%, 20%, 40%, 60%, 80% (detection depth subsample) + 100% (full testing).
Each depth: inference on subsampled z-scored matrices, then 1000× bootstrap.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

CFCRAFT = Path("/fh/working/sun_w/saishi_cui/cfCraft")
MATHIOS = CFCRAFT / "Mathios"
SCRIPTS = MATHIOS / "scripts"
FIG4_DIR = CFCRAFT / "mFrag_Figures" / "Figure4"
OUT_DIR_DEFAULT = CFCRAFT / "mFrag_Figures" / "Figure6"
FINAL_DATA_24 = MATHIOS / "Final_data_24"
DETECTION_ROOT = FINAL_DATA_24 / "detection depth"
FD_DIR = MATHIOS / "FDConv"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(FIG4_DIR) not in sys.path:
    sys.path.insert(0, str(FIG4_DIR))

from cnn_sweep4_frag3ch_lr_fine_wd3e6 import _build_test_arrays  # noqa: E402
from cnn_sweep_preset29_frag3ch_dual_beta_z import (  # noqa: E402
    Preset29ThreeChannelCNN,
    ThreeChannelZscoreDataset,
    build_preset29_cfg,
    collect_paths,
    try_three_channel_model,
    _spatial_ok_three_ch,
)
from fdconv_final24_frag_meth3ch_new import (  # noqa: E402
    FINAL_BATCH_SIZE,
    FINAL_EPOCHS,
    FINAL_LR,
    FINAL_WD,
)
from figure4_plot_titles import bootstrap_plot_title  # noqa: E402
from sweep_cnn_round10_r49_zscored_ep75_ep100 import _predict_loader_flexible  # noqa: E402

DEPTHS = ("0.1%", "1%", "5%", "10%", "20%", "40%", "60%", "80%", "100%")
SUBSAMPLED_DEPTHS = ("0.1%", "1%", "5%", "10%", "20%", "40%", "60%", "80%")
MODELS = (
    {
        "key": "fraglength",
        "label": "mFrag-Net (Frag length)",
        "main_subdir": "testing_fragment_length_zscored",
        "weights": FD_DIR / "frag_meth3ch_new" / "runs" / "rep_031_weights.pt",
        "full_bootstrap": FD_DIR / "frag_meth3ch_new_rep031_bootstrap_compare" / "bootstrap_draws.csv",
        "bootstrap_seed_base": 73,
        "color": "#9467bd",
    },
    {
        "key": "motif",
        "label": "mFrag-Net (Motif)",
        "main_subdir": "testing_end4motif_zscored",
        "weights": FD_DIR / "motif_meth3ch_new" / "runs" / "rep_006_weights.pt",
        "full_bootstrap": FD_DIR / "motif_meth3ch_new_rep006_bootstrap_compare" / "bootstrap_draws.csv",
        "bootstrap_seed_base": 48,
        "color": "#ff7f0e",
    },
)

FULL_TEST_MAIN = {
    "testing_fragment_length_zscored": FINAL_DATA_24 / "testing" / "testing_fragment_length_zscored",
    "testing_end4motif_zscored": FINAL_DATA_24 / "testing" / "testing_end4motif_zscored",
}
FULL_TEST_H = FINAL_DATA_24 / "testing" / "testing_DNAMethylation_HealthyRef_zscored"
FULL_TEST_L = FINAL_DATA_24 / "testing" / "testing_DNAMethylation_LungcancerRef_zscored"

FONT_BASE = 14
FONT_TITLE = 17
FONT_LABEL = 15
FONT_TICK = 13
FONT_LEGEND = 11
BOX_SPAN = 0.42
BOX_WIDTH = 0.11
BOX_XPAD = 0.24
METRIC_CENTERS = (1.0, 2.0)


def _depth_dirs(depth: str, main_subdir: str) -> tuple[Path, Path, Path]:
    if depth == "100%":
        base = FINAL_DATA_24 / "testing"
    else:
        base = DETECTION_ROOT / depth
    return base / main_subdir, base / "testing_DNAMethylation_HealthyRef_zscored", base / "testing_DNAMethylation_LungcancerRef_zscored"


def _expected_test_paths(main_subdir: str) -> list[Path]:
    return collect_paths(FULL_TEST_MAIN[main_subdir] if main_subdir in FULL_TEST_MAIN else FULL_TEST_MAIN["testing_fragment_length_zscored"])


def _verify_depth_complete(depth: str, main_subdir: str, test_paths: list[Path], n_required: int) -> None:
    main, h, l = _depth_dirs(depth, main_subdir)
    missing = []
    for p in test_paths:
        stem = p.stem
        for d in (main, h, l):
            if not (d / f"{stem}.csv").is_file():
                missing.append(f"{depth}:{stem}")
                break
    if len(test_paths) - len(missing) < n_required:
        raise RuntimeError(
            f"{depth} incomplete: {len(test_paths) - len(missing)}/{n_required} samples ready "
            f"({len(missing)} missing). Wait for detection depth build to finish."
        )


def _load_model(weights_path: Path, device: torch.device) -> Preset29ThreeChannelCNN:
    cfg = replace(
        build_preset29_cfg(FINAL_EPOCHS),
        lr=float(FINAL_LR),
        weight_decay=float(FINAL_WD),
        batch_size=int(FINAL_BATCH_SIZE),
    )
    if not _spatial_ok_three_ch(cfg) or not try_three_channel_model(cfg):
        raise RuntimeError("invalid model cfg")
    ckpt = torch.load(weights_path, map_location=device)
    model = Preset29ThreeChannelCNN(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _predict_depth(
    *,
    depth: str,
    main_subdir: str,
    weights_path: Path,
    test_paths: list[Path],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    main, h, l = _depth_dirs(depth, main_subdir)
    paths_ok = [p for p in test_paths if (main / f"{p.stem}.csv").is_file()]
    ds = ThreeChannelZscoreDataset(main, h, l, paths_ok)
    loader = DataLoader(ds, batch_size=FINAL_BATCH_SIZE, shuffle=False, num_workers=0)
    model = _load_model(weights_path, device)
    y_true, prob = _predict_loader_flexible(model, loader, device)
    _, _, labels_ref = _build_test_arrays(test_paths)
    y = np.asarray(y_true, dtype=np.int32)
    if len(y) != len(labels_ref):
        raise RuntimeError(f"{depth}: predicted n={len(y)} != expected {len(labels_ref)}")
    return y, np.asarray(prob, dtype=np.float64)


def _bootstrap_draws(y: np.ndarray, prob: np.ndarray, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs: list[float] = []
    praucs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        p_b = prob[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aucs.append(float(roc_auc_score(y_b, p_b)))
        praucs.append(float(average_precision_score(y_b, p_b)))
    return pd.DataFrame(
        {
            "bootstrap_idx": np.arange(len(aucs)),
            "test_auc": aucs,
            "test_prauc": praucs,
        }
    )


def _run_inference_and_bootstrap(
    model_cfg: dict,
    *,
    test_paths: list[Path],
    n_boot: int,
    n_required: int,
    cache_dir: Path,
    device: torch.device,
    force: bool,
    depths: tuple[str, ...] = DEPTHS,
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for depth in depths:
        draws_path = cache_dir / model_cfg["key"] / depth / "bootstrap_draws.csv"
        if depth == "100%" and model_cfg["full_bootstrap"].is_file() and not force:
            df = pd.read_csv(model_cfg["full_bootstrap"])
            out[depth] = {
                "auc": df["test_auc"].to_numpy(dtype=np.float64),
                "prauc": df["test_prauc"].to_numpy(dtype=np.float64),
            }
            continue
        if draws_path.is_file() and not force:
            df = pd.read_csv(draws_path)
            out[depth] = {
                "auc": df["test_auc"].to_numpy(dtype=np.float64),
                "prauc": df["test_prauc"].to_numpy(dtype=np.float64),
            }
            continue

        _verify_depth_complete(depth, model_cfg["main_subdir"], test_paths, n_required)
        y, prob = _predict_depth(
            depth=depth,
            main_subdir=model_cfg["main_subdir"],
            weights_path=model_cfg["weights"],
            test_paths=test_paths,
            device=device,
        )
        seed = int(model_cfg["bootstrap_seed_base"] + DEPTHS.index(depth) * 997)
        draws = _bootstrap_draws(y, prob, n_boot, seed)
        draws_path.parent.mkdir(parents=True, exist_ok=True)
        draws.to_csv(draws_path, index=False)
        np.savez_compressed(
            draws_path.parent / "test_predictions.npz",
            labels=y.astype(np.int32),
            probs=prob.astype(np.float64),
            depth=depth,
        )
        out[depth] = {
            "auc": draws["test_auc"].to_numpy(dtype=np.float64),
            "prauc": draws["test_prauc"].to_numpy(dtype=np.float64),
        }
        print(f"Wrote {draws_path} (median AUC={np.median(out[depth]['auc']):.4f})", flush=True)

    for depth in DEPTHS:
        if depth in out:
            continue
        draws_path = cache_dir / model_cfg["key"] / depth / "bootstrap_draws.csv"
        if depth == "100%" and model_cfg["full_bootstrap"].is_file():
            df = pd.read_csv(model_cfg["full_bootstrap"])
        elif draws_path.is_file():
            df = pd.read_csv(draws_path)
        else:
            raise FileNotFoundError(f"Missing cached bootstrap for {model_cfg['key']} {depth}: {draws_path}")
        out[depth] = {
            "auc": df["test_auc"].to_numpy(dtype=np.float64),
            "prauc": df["test_prauc"].to_numpy(dtype=np.float64),
        }
    return out


def _plot_boxplot(
    data: dict[str, dict[str, np.ndarray]],
    *,
    model_label: str,
    color: str,
    out_png: Path,
    out_pdf: Path,
    plot_title: str,
    ylim: tuple[float, float],
    dpi: int,
) -> None:
    metric_keys = ("auc", "prauc")
    metric_labels = ("Testing AUC", "Testing PRAUC")
    group_centers = np.array(METRIC_CENTERS, dtype=float)
    offsets = np.linspace(-BOX_SPAN / 2, BOX_SPAN / 2, len(DEPTHS))
    pos_map: dict[tuple[str, str], float] = {}

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

    box_data: list[np.ndarray] = []
    for mi, mkey in enumerate(metric_keys):
        gc = group_centers[mi]
        for gi, depth in enumerate(DEPTHS):
            pos_map[(mkey, depth)] = gc + offsets[gi]
            box_data.append(data[depth][mkey])

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    bp = ax.boxplot(
        box_data,
        positions=[pos_map[(mk, d)] for mk in metric_keys for d in DEPTHS],
        widths=BOX_WIDTH,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 2.2},
        whiskerprops={"color": "black", "linewidth": 1.5},
        capprops={"color": "black", "linewidth": 1.5},
        boxprops={"edgecolor": "black", "linewidth": 1.5},
    )
    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(0.35)

    rng = np.random.default_rng(7)
    y_lo, y_hi = ylim
    for mkey in metric_keys:
        for depth in DEPTHS:
            pos = pos_map[(mkey, depth)]
            pts = np.asarray(data[depth][mkey], dtype=np.float64)
            pts = pts[(pts >= y_lo) & (pts <= y_hi) & np.isfinite(pts)]
            if pts.size == 0:
                continue
            k = min(pts.size, 60)
            if k < pts.size:
                pts = pts[rng.choice(pts.size, size=k, replace=False)]
            r, g, b = to_rgb(color)
            light = (0.55 + 0.45 * r, 0.55 + 0.45 * g, 0.55 + 0.45 * b)
            jitter = rng.uniform(-0.03, 0.03, size=pts.size)
            ax.scatter(
                np.full_like(pts, pos) + jitter,
                pts,
                s=14,
                color=light,
                alpha=0.55,
                linewidths=0.0,
                zorder=3,
            )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(metric_labels, fontweight="bold")
    depth_ticks = [(group_centers[0] + offsets[i], DEPTHS[i]) for i in range(len(DEPTHS))]
    depth_ticks += [(group_centers[1] + offsets[i], DEPTHS[i]) for i in range(len(DEPTHS))]
    for x, lbl in depth_ticks:
        ax.text(x, y_lo - 0.06 * (y_hi - y_lo), lbl, ha="center", va="top", fontsize=9, fontweight="bold")

    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title(f"{plot_title}\n{model_label}", fontsize=FONT_TITLE, fontweight="bold")
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="both", which="both", width=1.2, labelsize=FONT_TICK)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

    left = group_centers[0] + offsets[0] - BOX_WIDTH / 2 - BOX_XPAD
    right = group_centers[1] + offsets[-1] + BOX_WIDTH / 2 + BOX_XPAD
    ax.set_xlim(left, right)

    ax.legend(
        handles=[Patch(facecolor=color, edgecolor="black", linewidth=1.2, alpha=0.35, label=model_label)],
        loc="upper right",
        framealpha=0.95,
        edgecolor="0.75",
        prop={"size": FONT_LEGEND, "weight": "bold"},
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def _plot_combined_depth_metric(
    frag_data: dict[str, np.ndarray],
    motif_data: dict[str, np.ndarray],
    *,
    metric_label: str,
    out_png: Path,
    out_pdf: Path,
    plot_title: str,
    ylim: tuple[float, float],
    dpi: int,
) -> None:
    """One figure: x = detection depth; grouped boxplots for Frag length vs Motif."""
    positions = np.arange(len(DEPTHS), dtype=float) + 1.0
    width = 0.32
    frag_pos = positions - width / 2
    motif_pos = positions + width / 2
    colors = {"frag": MODELS[0]["color"], "motif": MODELS[1]["color"]}

    mpl.rcParams.update(
        {
            "font.size": FONT_BASE,
            "axes.titlesize": FONT_TITLE,
            "axes.labelsize": FONT_LABEL,
            "legend.fontsize": FONT_LEGEND,
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    bp_frag = ax.boxplot(
        [frag_data[d] for d in DEPTHS],
        positions=frag_pos,
        widths=width * 0.88,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 2.2},
    )
    bp_motif = ax.boxplot(
        [motif_data[d] for d in DEPTHS],
        positions=motif_pos,
        widths=width * 0.88,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 2.2},
    )
    for box in bp_frag["boxes"]:
        box.set_facecolor(colors["frag"])
        box.set_alpha(0.35)
    for box in bp_motif["boxes"]:
        box.set_facecolor(colors["motif"])
        box.set_alpha(0.35)

    ax.set_xticks(positions)
    ax.set_xticklabels(DEPTHS, fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_xlabel("Detection depth (cfDNA fragment fraction)", fontweight="bold")
    ax.set_title(f"{plot_title}\n{metric_label}", fontweight="bold")
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(
        handles=[
            Patch(facecolor=colors["frag"], edgecolor="black", linewidth=1.2, alpha=0.35, label=MODELS[0]["label"]),
            Patch(facecolor=colors["motif"], edgecolor="black", linewidth=1.2, alpha=0.35, label=MODELS[1]["label"]),
        ],
        loc="lower right",
        framealpha=0.95,
        edgecolor="0.75",
        prop={"size": FONT_LEGEND, "weight": "bold"},
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def _parse_depths(s: str) -> tuple[str, ...]:
    depths = tuple(x.strip() for x in s.split(",") if x.strip())
    unknown = [d for d in depths if d not in DEPTHS]
    if unknown:
        raise SystemExit(f"Unknown depth(s): {unknown}. Valid: {', '.join(DEPTHS)}")
    order = {d: i for i, d in enumerate(DEPTHS)}
    return tuple(sorted(depths, key=lambda d: order[d]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--cache-dir", type=Path, default=OUT_DIR_DEFAULT / "detection_depth_bootstrap")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--n-required", type=int, default=431)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--ylim", type=float, nargs=2, default=(0.0, 1.2))
    ap.add_argument(
        "--depths",
        type=str,
        default=",".join(DEPTHS),
        help=f"Comma-separated depths to infer (subset of {', '.join(DEPTHS)}).",
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--inference-only", action="store_true", help="Skip all figure outputs.")
    args = ap.parse_args()
    run_depths = _parse_depths(args.depths)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    frag_paths = collect_paths(FULL_TEST_MAIN["testing_fragment_length_zscored"])
    motif_paths = collect_paths(FULL_TEST_MAIN["testing_end4motif_zscored"])
    sample_ids = [p.stem for p in frag_paths]
    plot_title = bootstrap_plot_title("overall", "All stages", sample_ids)

    summary_rows = []
    all_model_data: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for model_cfg in MODELS:
        test_paths = frag_paths if model_cfg["key"] == "fraglength" else motif_paths
        if args.plot_only:
            data = {}
            for depth in DEPTHS:
                if depth == "100%":
                    df = pd.read_csv(model_cfg["full_bootstrap"])
                else:
                    df = pd.read_csv(args.cache_dir / model_cfg["key"] / depth / "bootstrap_draws.csv")
                data[depth] = {
                    "auc": df["test_auc"].to_numpy(dtype=np.float64),
                    "prauc": df["test_prauc"].to_numpy(dtype=np.float64),
                }
        else:
            data = _run_inference_and_bootstrap(
                model_cfg,
                test_paths=test_paths,
                n_boot=args.n_bootstrap,
                n_required=args.n_required,
                cache_dir=args.cache_dir,
                device=device,
                force=args.force,
                depths=run_depths,
            )
        all_model_data[model_cfg["key"]] = data

        for depth in DEPTHS:
            for mkey in ("auc", "prauc"):
                arr = data[depth][mkey]
                summary_rows.append(
                    {
                        "model_key": model_cfg["key"],
                        "model_label": model_cfg["label"],
                        "depth": depth,
                        "metric": mkey,
                        "n": len(arr),
                        "median": float(np.median(arr)),
                        "q25": float(np.percentile(arr, 25)),
                        "q75": float(np.percentile(arr, 75)),
                    }
                )

        if args.inference_only:
            continue

        slug = model_cfg["key"]
        _plot_boxplot(
            data,
            model_label=model_cfg["label"],
            color=model_cfg["color"],
            out_png=args.out_dir / f"figure6_detection_depth_{slug}_bootstrap_boxplot.png",
            out_pdf=args.out_dir / f"figure6_detection_depth_{slug}_bootstrap_boxplot.pdf",
            plot_title=plot_title,
            ylim=tuple(args.ylim),
            dpi=args.dpi,
        )

    if args.inference_only:
        pd.DataFrame(summary_rows).to_csv(
            args.cache_dir / "figure6_detection_depth_bootstrap_summary_partial.csv",
            index=False,
        )
        print(pd.DataFrame(summary_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        return 0

    for mkey, mlabel, fname in (
        ("auc", "Testing AUC", "testing_auc"),
        ("prauc", "Testing PRAUC", "testing_prauc"),
    ):
        _plot_combined_depth_metric(
            {d: all_model_data["fraglength"][d][mkey] for d in DEPTHS},
            {d: all_model_data["motif"][d][mkey] for d in DEPTHS},
            metric_label=mlabel,
            out_png=args.out_dir / f"figure6_detection_depth_{fname}_bootstrap_boxplot.png",
            out_pdf=args.out_dir / f"figure6_detection_depth_{fname}_bootstrap_boxplot.pdf",
            plot_title=plot_title,
            ylim=tuple(args.ylim),
            dpi=args.dpi,
        )

    pd.DataFrame(summary_rows).to_csv(
        args.out_dir / "figure6_detection_depth_bootstrap_summary.csv",
        index=False,
    )
    (args.cache_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "depths": list(DEPTHS),
                "n_bootstrap": args.n_bootstrap,
                "n_required": args.n_required,
                "models": [m["key"] for m in MODELS],
            },
            indent=2,
        )
    )
    print(pd.DataFrame(summary_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
