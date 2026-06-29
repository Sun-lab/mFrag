#!/usr/bin/env python3
"""
Window x motif-bin IG heatmap (top 30 windows) for mFrag-Net (End 4-mer motif, 3ch).

All testing cancer vs all testing healthy; motif_meth3ch_new rep_006.
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
OUT_DIR = CFCRAFT / "mFrag_Figures" / "Figure6"
FINAL_DATA_24 = MATHIOS / "Final_data_24"
MOTIF_TEST = FINAL_DATA_24 / "testing" / "testing_end4motif_zscored"
H_TEST = FINAL_DATA_24 / "testing" / "testing_DNAMethylation_HealthyRef_zscored"
L_TEST = FINAL_DATA_24 / "testing" / "testing_DNAMethylation_LungcancerRef_zscored"
WEIGHTS_PT = MATHIOS / "FDConv" / "motif_meth3ch_new" / "runs" / "rep_006_weights.pt"
CACHE_NPZ = OUT_DIR / "figure6_motif_mfrag_ig_3ch_window_bin_testing_cache.npz"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cnn_sweep_preset29_frag3ch_dual_beta_z import (  # noqa: E402
    Preset29ThreeChannelCNN,
    ThreeChannelZscoreDataset,
    build_preset29_cfg,
    collect_paths,
)
from fdconv_final24_motif_meth3ch_new import FINAL_EPOCHS, FINAL_LR, FINAL_WD  # noqa: E402
from train_cnn_fragment_histogram import N_LEN_BINS  # noqa: E402

IG_STEPS = 32
TOP_N = 30
CBAR_LABEL = "|mean cancer IG − mean healthy IG|"


def _is_healthy(stem: str) -> bool:
    return stem.endswith("_healthy")


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


def _load_3ch(motif_path: Path) -> np.ndarray:
    ds = ThreeChannelZscoreDataset(MOTIF_TEST, H_TEST, L_TEST, [motif_path])
    x, _ = ds[0]
    return x.numpy()


def _load_model(device: torch.device) -> Preset29ThreeChannelCNN:
    cfg = replace(
        build_preset29_cfg(FINAL_EPOCHS),
        lr=float(FINAL_LR),
        weight_decay=float(FINAL_WD),
    )
    model = Preset29ThreeChannelCNN(cfg).to(device)
    ckpt = torch.load(WEIGHTS_PT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _motif_bin_labels(sample_path: Path) -> list[str]:
    cols = pd.read_csv(sample_path, nrows=0).columns.tolist()
    if len(cols) != N_LEN_BINS + 1:
        raise SystemExit(f"Expected {N_LEN_BINS} motif bins, got {len(cols) - 1} in {sample_path}")
    return cols[1:]


def _plot_heatmap(
    mat: np.ndarray,
    win_ids: list[str],
    motif_labels: list[str],
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
    ax.set_xlabel("End 4-mer motif bin", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(np.arange(N_LEN_BINS))
    ax.set_xticklabels(
        motif_labels,
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

    paths = collect_paths(MOTIF_TEST)
    healthy_paths = [p for p in paths if _is_healthy(p.stem)]
    n_healthy, n_cancer = len(healthy_paths), len(paths) - len(healthy_paths)
    print(f"testing: {n_cancer} cancer, {n_healthy} healthy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    motif_labels = _motif_bin_labels(healthy_paths[0])

    cache_path = args.out_dir / CACHE_NPZ.name
    if cache_path.is_file() and not args.recompute:
        print(f"Loading cache {cache_path}")
        z = np.load(cache_path, allow_pickle=True)
        labels = z["labels"]
        mats = z["window_bin_ig"]
        win_ids = z["window_ids"].tolist()
    else:
        model = _load_model(device)
        print("Building healthy baseline...")
        x_h = np.stack([_load_3ch(p) for p in healthy_paths], axis=0)
        baseline = torch.from_numpy(x_h.mean(axis=0)).float()
        win_ids = pd.read_csv(healthy_paths[0]).iloc[:, 0].astype(str).tolist()

        mats_list: list[np.ndarray] = []
        labels_list: list[int] = []
        t0 = time.time()
        for i, path in enumerate(paths):
            x = torch.from_numpy(_load_3ch(path)).float()
            ig = _integrated_gradients(model, x, baseline, device=device)
            mats_list.append(_attr_window_bin(ig))
            labels_list.append(0 if _is_healthy(path.stem) else 1)
            if (i + 1) % 25 == 0 or i + 1 == len(paths):
                print(f"  IG {i + 1}/{len(paths)} ({time.time() - t0:.0f}s)", flush=True)

        labels = np.array(labels_list, dtype=np.int32)
        mats = np.stack(mats_list, axis=0)
        np.savez(cache_path, labels=labels, window_bin_ig=mats, window_ids=np.array(win_ids, dtype=object))
        print(f"Wrote cache {cache_path}")

    cancer_mask = labels == 1
    healthy_mask = labels == 0
    contrast = np.abs(mats[cancer_mask].mean(axis=0) - mats[healthy_mask].mean(axis=0))

    top_idx = np.argsort(-contrast.sum(axis=1))[: args.top_n]
    top_idx.sort()
    zoom = contrast[top_idx, :]
    zoom_wids = [win_ids[i] for i in top_idx]

    title = (
        "mFrag-Net (End 4-mer motif) Integrated Gradients\n"
        f"(all testing: {n_cancer} cancer vs {n_healthy} healthy)"
    )
    stem = "figure6_motif_mfrag_ig_window_bin_contrast_top30_windows"
    _plot_heatmap(
        zoom,
        zoom_wids,
        motif_labels,
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
                "argmax_motif": motif_labels[int(contrast[wi].argmax())],
            }
        )
    csv_path = args.out_dir / f"{stem}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")
    print(pd.DataFrame(rows).head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
