#!/usr/bin/env python3
"""
Compare window-level IG contrast (cancer vs healthy) between frag_meth3ch and fragonly.

All testing samples: cancer vs healthy IG contrast per model, then
|contrast_3ch - contrast_1ch| per genomic window.
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
FRAG_TEST = FINAL_DATA_24 / "testing" / "testing_fragment_length_zscored"
H_TEST = FINAL_DATA_24 / "testing" / "testing_DNAMethylation_HealthyRef_zscored"
L_TEST = FINAL_DATA_24 / "testing" / "testing_DNAMethylation_LungcancerRef_zscored"
WEIGHTS_3CH = MATHIOS / "FDConv" / "frag_meth3ch_new" / "runs" / "rep_031_weights.pt"
WEIGHTS_1CH = MATHIOS / "FDConv" / "fragonly_new" / "runs" / "rep_031_weights.pt"
CACHE_NPZ = OUT_DIR / "figure6_mfrag_ig_testing_all_samples_cache.npz"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cnn_sweep_preset29_frag3ch_dual_beta_z import (  # noqa: E402
    Preset29ThreeChannelCNN,
    ThreeChannelZscoreDataset,
    build_preset29_cfg,
    collect_paths,
)
from fdconv_final24_frag_meth3ch_new import FINAL_EPOCHS, FINAL_LR, FINAL_WD  # noqa: E402
from sweep_cnn_fulltrain_testing_roc import FlexibleTwoStageCNN  # noqa: E402
from train_cnn_fragment_histogram import FragmentWindowsDataset  # noqa: E402

IG_STEPS = 32
TOP_N = 30


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
    model: torch.nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    *,
    device: torch.device,
    steps: int = IG_STEPS,
) -> torch.Tensor:
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

    return (delta * grad_accum / steps).detach().cpu()


def _window_profile(ig: np.ndarray) -> np.ndarray:
    if ig.ndim == 3:
        prof = np.abs(ig).sum(axis=(1, 2))
    else:
        prof = np.abs(ig).sum(axis=1)
    total = prof.sum()
    return prof / total if total > 0 else prof


def _load_3ch(path: Path) -> np.ndarray:
    ds = ThreeChannelZscoreDataset(FRAG_TEST, H_TEST, L_TEST, [path])
    x, _ = ds[0]
    return x.numpy()


def _load_1ch(path: Path) -> np.ndarray:
    ds = FragmentWindowsDataset([path])
    x, _ = ds[0]
    return x.numpy()


def _load_models(device: torch.device) -> tuple[Preset29ThreeChannelCNN, FlexibleTwoStageCNN]:
    cfg = replace(
        build_preset29_cfg(FINAL_EPOCHS),
        lr=float(FINAL_LR),
        weight_decay=float(FINAL_WD),
    )
    m3 = Preset29ThreeChannelCNN(cfg).to(device)
    m1 = FlexibleTwoStageCNN(cfg).to(device)
    ck3 = torch.load(WEIGHTS_3CH, map_location=device, weights_only=False)
    ck1 = torch.load(WEIGHTS_1CH, map_location=device, weights_only=False)
    m3.load_state_dict(ck3["state_dict"])
    m1.load_state_dict(ck1["state_dict"])
    m3.eval()
    m1.eval()
    return m3, m1


def _compute_all_ig(
    paths: list[Path],
    m3: Preset29ThreeChannelCNN,
    m1: FlexibleTwoStageCNN,
    baseline_3ch: torch.Tensor,
    baseline_1ch: torch.Tensor,
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    stems: list[str] = []
    labels: list[int] = []
    profs_3ch: list[np.ndarray] = []
    profs_1ch: list[np.ndarray] = []
    t0 = time.time()

    for i, path in enumerate(paths):
        stem = path.stem
        x3 = torch.from_numpy(_load_3ch(path)).float()
        x1 = torch.from_numpy(_load_1ch(path)).float()
        ig3 = _integrated_gradients(m3, x3, baseline_3ch, device=device)
        ig1 = _integrated_gradients(m1, x1, baseline_1ch, device=device)
        profs_3ch.append(_window_profile(ig3.numpy()))
        profs_1ch.append(_window_profile(ig1.numpy()))
        stems.append(stem)
        labels.append(0 if _is_healthy(stem) else 1)
        if (i + 1) % 25 == 0 or i + 1 == len(paths):
            elapsed = time.time() - t0
            print(f"  IG {i + 1}/{len(paths)} ({elapsed:.0f}s)", flush=True)

    return stems, np.array(labels, dtype=np.int32), np.stack(profs_3ch), np.stack(profs_1ch)


def _plot_top_windows(
    win_ids: list[str],
    diff: np.ndarray,
    top_idx: np.ndarray,
    *,
    n_cancer: int,
    n_healthy: int,
    out_png: Path,
    out_pdf: Path,
    dpi: int,
) -> None:
    labels = [_win_label(win_ids[i]) for i in top_idx]
    y = np.arange(len(top_idx))

    fig, ax = plt.subplots(figsize=(12, 13))
    colors = plt.cm.YlOrRd(diff[top_idx] / max(diff[top_idx].max(), 1e-12))
    ax.barh(y, diff[top_idx], color=colors, alpha=0.95)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel(
        "|mean cancer IG − mean healthy IG| "
        "(with DNA methyl − without DNA methyl)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(f"Top {len(top_idx)} windows by IG contrast difference", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="x")
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")
        lbl.set_fontsize(11)

    fig.suptitle(
        "mFrag-Net Integrated Gradients\n"
        "mFrag-Net (Frag length) vs mFrag-Net (Frag length without DNA Methyl)\n"
        f"(all testing: {n_cancer} cancer vs {n_healthy} healthy; baseline = mean healthy)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.subplots_adjust(top=0.90)
    fig.tight_layout()
    for out in (out_png, out_pdf):
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--top-n", type=int, default=TOP_N)
    ap.add_argument("--recompute", action="store_true", help="Ignore cached IG profiles")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = collect_paths(FRAG_TEST)
    healthy_paths = [p for p in paths if _is_healthy(p.stem)]
    cancer_paths = [p for p in paths if not _is_healthy(p.stem)]
    n_healthy, n_cancer = len(healthy_paths), len(cancer_paths)
    print(f"testing: {n_cancer} cancer, {n_healthy} healthy ({len(paths)} total)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    cache_path = args.out_dir / CACHE_NPZ.name
    if cache_path.is_file() and not args.recompute:
        print(f"Loading cached IG profiles from {cache_path}")
        z = np.load(cache_path, allow_pickle=True)
        labels = z["labels"]
        profs_3ch = z["profs_3ch"]
        profs_1ch = z["profs_1ch"]
        win_ids = z["window_ids"].tolist()
    else:
        m3, m1 = _load_models(device)
        print("Building healthy baseline (mean over all testing healthy)...")
        x3_h = np.stack([_load_3ch(p) for p in healthy_paths], axis=0)
        x1_h = np.stack([_load_1ch(p) for p in healthy_paths], axis=0)
        baseline_3ch = torch.from_numpy(x3_h.mean(axis=0)).float()
        baseline_1ch = torch.from_numpy(x1_h.mean(axis=0)).float()
        win_ids = pd.read_csv(healthy_paths[0]).iloc[:, 0].astype(str).tolist()

        print("Computing IG for all testing samples...")
        _stems, labels, profs_3ch, profs_1ch = _compute_all_ig(
            paths, m3, m1, baseline_3ch, baseline_1ch, device
        )
        np.savez(
            cache_path,
            labels=labels,
            profs_3ch=profs_3ch,
            profs_1ch=profs_1ch,
            window_ids=np.array(win_ids, dtype=object),
        )
        print(f"Wrote cache {cache_path}")

    cancer_mask = labels == 1
    healthy_mask = labels == 0

    mean_c3 = profs_3ch[cancer_mask].mean(axis=0)
    mean_h3 = profs_3ch[healthy_mask].mean(axis=0)
    mean_c1 = profs_1ch[cancer_mask].mean(axis=0)
    mean_h1 = profs_1ch[healthy_mask].mean(axis=0)

    contrast_3ch = np.abs(mean_c3 - mean_h3)
    contrast_1ch = np.abs(mean_c1 - mean_h1)
    diff = np.abs(contrast_3ch - contrast_1ch)

    rows = []
    for i, wid in enumerate(win_ids):
        rows.append(
            {
                "window_id": wid,
                "window_label": _win_label(wid),
                "contrast_3ch": float(contrast_3ch[i]),
                "contrast_1ch": float(contrast_1ch[i]),
                "abs_contrast_diff": float(diff[i]),
                "mean_ig_frac_3ch_cancer": float(mean_c3[i]),
                "mean_ig_frac_3ch_healthy": float(mean_h3[i]),
                "mean_ig_frac_1ch_cancer": float(mean_c1[i]),
                "mean_ig_frac_1ch_healthy": float(mean_h1[i]),
            }
        )
    rank_df = pd.DataFrame(rows).sort_values("abs_contrast_diff", ascending=False)
    stem = "figure6_mfrag_ig_frag_meth3ch_vs_fragonly_window_diff"
    csv_path = args.out_dir / f"{stem}.csv"
    rank_df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    top_idx = np.argsort(-diff)[: args.top_n]
    _plot_top_windows(
        win_ids,
        diff,
        top_idx,
        n_cancer=n_cancer,
        n_healthy=n_healthy,
        out_png=args.out_dir / f"{stem}_top{args.top_n}.png",
        out_pdf=args.out_dir / f"{stem}_top{args.top_n}.pdf",
        dpi=args.dpi,
    )

    print(f"\nTop {min(15, args.top_n)} windows by |contrast_3ch − contrast_1ch|:")
    print(rank_df.head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
