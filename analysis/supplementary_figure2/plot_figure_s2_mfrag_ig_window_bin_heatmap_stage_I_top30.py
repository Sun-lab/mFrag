#!/usr/bin/env python3
"""
Figure S2 — window x bin IG heatmap (top 30 windows) for mFrag-Net (Frag length, 3ch).

Testing Stage I lung cancer vs all testing healthy; reuses Figure6 IG cache when available.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

CFCRAFT = Path("/fh/working/sun_w/saishi_cui/cfCraft")
MATHIOS = CFCRAFT / "Mathios"
SCRIPTS = MATHIOS / "scripts"
OUT_DIR = CFCRAFT / "mFrag_Figures" / "Figure S2"
FIG6_DIR = CFCRAFT / "mFrag_Figures" / "Figure6"
FINAL_DATA_24 = MATHIOS / "Final_data_24"
FRAG_TEST = FINAL_DATA_24 / "testing" / "testing_fragment_length_zscored"
H_TEST = FINAL_DATA_24 / "testing" / "testing_DNAMethylation_HealthyRef_zscored"
L_TEST = FINAL_DATA_24 / "testing" / "testing_DNAMethylation_LungcancerRef_zscored"
META_PATH = MATHIOS / "metadata" / "validation_testing_sample_list.tsv"
WEIGHTS_3CH = MATHIOS / "FDConv" / "frag_meth3ch_new" / "runs" / "rep_031_weights.pt"
CACHE_NPZ = FIG6_DIR / "figure6_mfrag_ig_3ch_window_bin_testing_cache.npz"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cnn_sweep_preset29_frag3ch_dual_beta_z import (  # noqa: E402
    Preset29ThreeChannelCNN,
    ThreeChannelZscoreDataset,
    build_preset29_cfg,
    collect_paths,
)
from fdconv_final24_frag_meth3ch_new import FINAL_EPOCHS, FINAL_LR, FINAL_WD  # noqa: E402
from train_cnn_fragment_histogram import N_LEN_BINS  # noqa: E402

BIN_LABELS = [f"{100 + 5 * j}-{100 + 5 * (j + 1)}" for j in range(23)] + ["215-220"]
IG_STEPS = 32
TOP_N = 30
CBAR_LABEL = "|mean Stage I cancer IG − mean healthy IG|"


def _is_healthy(stem: str) -> bool:
    return stem.endswith("_healthy")


def _stage_group(stage: str) -> str | None:
    s = str(stage).strip().upper()
    if s == "I":
        return "stage_I"
    return None


def _cg_id(stem: str) -> str:
    return stem.replace("_healthy", "").replace("_lung_cancer", "")


def _load_stage_map() -> dict[str, str]:
    meta = pd.read_csv(META_PATH, sep="\t")
    return meta.set_index("sample_id")["Stage"].astype(str).to_dict()


def _win_label(wid: str) -> str:
    parts = wid.split("_")
    if len(parts) < 3:
        return wid
    chrom = parts[0].replace("chr", "Chr")
    s = int(parts[1]) // 1_000_000
    e = int(parts[2]) // 1_000_000
    return f"{chrom}:{s}-{e}Mb"


def _integrated_gradients(
    model: Preset29ThreeChannelCNN,
    x: torch.Tensor,
    baseline: torch.Tensor,
    *,
    device: torch.device,
    steps: int = IG_STEPS,
) -> np.ndarray:
    x = x.to(device)
    baseline = baseline.to(device)
    delta = x - baseline
    alphas = torch.linspace(0.0, 1.0, steps, device=device)
    grad_accum = torch.zeros_like(x)

    for alpha in alphas:
        x_interp = (baseline + alpha * delta).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logit = model(x_interp.unsqueeze(0)).squeeze()
        logit.backward()
        grad_accum += x_interp.grad.detach()

    ig = (delta * grad_accum / steps).detach().cpu().numpy()
    return ig


def _attr_window_bin(ig: np.ndarray) -> np.ndarray:
    return np.abs(ig).sum(axis=-1)


def _load_3ch(path: Path) -> np.ndarray:
    ds = ThreeChannelZscoreDataset(FRAG_TEST, H_TEST, L_TEST, [path])
    x, _ = ds[0]
    return x.numpy()


def _load_model(device: torch.device) -> Preset29ThreeChannelCNN:
    cfg = replace(
        build_preset29_cfg(FINAL_EPOCHS),
        lr=float(FINAL_LR),
        weight_decay=float(FINAL_WD),
    )
    model = Preset29ThreeChannelCNN(cfg).to(device)
    ckpt = torch.load(WEIGHTS_3CH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _build_masks(paths: list[Path], stage_map: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    healthy_mask = np.zeros(len(paths), dtype=bool)
    stage_i_mask = np.zeros(len(paths), dtype=bool)
    for i, path in enumerate(paths):
        stem = path.stem
        if _is_healthy(stem):
            healthy_mask[i] = True
        elif _stage_group(stage_map.get(_cg_id(stem), "")) == "stage_I":
            stage_i_mask[i] = True
    return stage_i_mask, healthy_mask


def _plot_heatmap(
    mat: np.ndarray,
    win_ids: list[str],
    *,
    title: str,
    out_png: Path,
    out_pdf: Path,
    dpi: int,
) -> None:
    n_rows = mat.shape[0]
    fig_h = max(9.0, n_rows * 0.28)
    fig, ax = plt.subplots(figsize=(12.5, fig_h))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Fragment length bin (bp)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(np.arange(N_LEN_BINS))
    ax.set_xticklabels(
        BIN_LABELS,
        rotation=45,
        fontsize=10,
        fontweight="bold",
        ha="right",
    )
    ax.set_yticks(list(range(n_rows)))
    ax.set_yticklabels([_win_label(w) for w in win_ids], fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label(CBAR_LABEL, fontsize=10, fontweight="bold")
    cbar.ax.tick_params(labelsize=10, colors="black")
    for t in cbar.ax.get_yticklabels():
        t.set_fontweight("bold")
        t.set_color("black")
    fig.subplots_adjust(left=0.22, right=0.92, top=0.94, bottom=0.16)
    for out in (out_png, out_pdf):
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--top-n", type=int, default=TOP_N)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = collect_paths(FRAG_TEST)
    stage_map = _load_stage_map()
    stage_i_mask, healthy_mask = _build_masks(paths, stage_map)
    n_stage_i = int(stage_i_mask.sum())
    n_healthy = int(healthy_mask.sum())
    if n_stage_i == 0:
        raise SystemExit("No Stage I testing cancer samples found")
    if n_healthy == 0:
        raise SystemExit("No testing healthy samples found")
    print(f"testing Stage I: {n_stage_i} cancer vs {n_healthy} healthy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    if CACHE_NPZ.is_file() and not args.recompute:
        print(f"Loading cache {CACHE_NPZ}")
        z = np.load(CACHE_NPZ, allow_pickle=True)
        mats = z["window_bin_ig"]
        win_ids = z["window_ids"].tolist()
        if mats.shape[0] != len(paths):
            raise SystemExit(
                f"Cache sample count {mats.shape[0]} != path count {len(paths)}; use --recompute"
            )
    else:
        model = _load_model(device)
        healthy_paths = [p for p in paths if _is_healthy(p.stem)]
        print("Building healthy baseline...")
        x_h = np.stack([_load_3ch(p) for p in healthy_paths], axis=0)
        baseline = torch.from_numpy(x_h.mean(axis=0)).float()
        win_ids = pd.read_csv(healthy_paths[0]).iloc[:, 0].astype(str).tolist()

        mats_list: list[np.ndarray] = []
        t0 = time.time()
        for i, path in enumerate(paths):
            x = torch.from_numpy(_load_3ch(path)).float()
            ig = _integrated_gradients(model, x, baseline, device=device)
            mats_list.append(_attr_window_bin(ig))
            if (i + 1) % 25 == 0 or i + 1 == len(paths):
                print(f"  IG {i + 1}/{len(paths)} ({time.time() - t0:.0f}s)", flush=True)
        mats = np.stack(mats_list, axis=0)

    contrast = np.abs(mats[stage_i_mask].mean(axis=0) - mats[healthy_mask].mean(axis=0))

    top_idx = np.argsort(-contrast.sum(axis=1))[: args.top_n]
    top_idx.sort()
    zoom = contrast[top_idx, :]
    zoom_wids = [win_ids[i] for i in top_idx]

    title = (
        "mFrag-Net (Frag length) Integrated Gradients\n"
        f"(testing Stage I: {n_stage_i} cancer vs {n_healthy} healthy)"
    )
    stem = "figure_s2_mfrag_ig_window_bin_contrast_stage_I_top30_windows"
    _plot_heatmap(
        zoom,
        zoom_wids,
        title=title,
        out_png=args.out_dir / f"{stem}.png",
        out_pdf=args.out_dir / f"{stem}.pdf",
        dpi=args.dpi,
    )

    rows = []
    for rank, wi in enumerate(top_idx, start=1):
        rows.append(
            {
                "rank": rank,
                "window_id": win_ids[wi],
                "window_label": _win_label(win_ids[wi]),
                "total_contrast": float(contrast[wi].sum()),
                "max_bin_contrast": float(contrast[wi].max()),
                "argmax_bin": BIN_LABELS[int(contrast[wi].argmax())],
            }
        )
    csv_path = args.out_dir / f"{stem}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")
    print(pd.DataFrame(rows).head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
