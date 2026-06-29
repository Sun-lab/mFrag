#!/usr/bin/env python3
"""
Integrated Gradients (IG) attribution heatmap for barplot-6 mFrag-Net testing samples.

Model: Preset29ThreeChannelCNN (frag_meth3ch_new, rep_031).
Rows = genomic windows, columns = 24 fragment-length bins.
Dark color = |mean(cancer IG) - mean(healthy IG)| (larger Stage II vs healthy contrast).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

CFCRAFT = Path("/fh/working/sun_w/saishi_cui/cfCraft")
MATHIOS = CFCRAFT / "Mathios"
SCRIPTS = MATHIOS / "scripts"
OUT_DIR = CFCRAFT / "mFrag_Figures" / "Figure5"
BARPLOT_CSV = OUT_DIR / "figure5_frag_length_stage_II_mfragnet_top2_vs_fp_healthy_predicted_prob_barplot.csv"
FINAL_DATA_24 = MATHIOS / "Final_data_24"
FRAG_TEST = FINAL_DATA_24 / "testing" / "testing_fragment_length_zscored"
H_TEST = FINAL_DATA_24 / "testing" / "testing_DNAMethylation_HealthyRef_zscored"
L_TEST = FINAL_DATA_24 / "testing" / "testing_DNAMethylation_LungcancerRef_zscored"
WEIGHTS_PT = MATHIOS / "FDConv" / "frag_meth3ch_new" / "runs" / "rep_031_weights.pt"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cnn_sweep_preset29_frag3ch_dual_beta_z import (  # noqa: E402
    Preset29ThreeChannelCNN,
    ThreeChannelZscoreDataset,
    build_preset29_cfg,
)
from fdconv_final24_frag_meth3ch_new import FINAL_EPOCHS, FINAL_LR, FINAL_WD  # noqa: E402
from train_cnn_fragment_histogram import N_LEN_BINS, N_WINDOWS  # noqa: E402

BIN_LABELS = [f"{100 + 5 * j}-{100 + 5 * (j + 1)}" for j in range(23)] + ["215-220"]
IG_STEPS = 32
TOP_ZOOM_WINDOWS = 30


def _ig_main_title(targets: pd.DataFrame) -> str:
    cancer_ids = ", ".join(
        targets.loc[targets["group"] != "Healthy", "short_id"].astype(str).tolist()
    )
    healthy_ids = ", ".join(
        targets.loc[targets["group"] == "Healthy", "short_id"].astype(str).tolist()
    )
    return (
        "mFrag-Net (Frag length) Integrated Gradients\n"
        f"(Stage II lung cancer: {cancer_ids}) vs (healthy: {healthy_ids})"
    )


def _load_targets() -> pd.DataFrame:
    df = pd.read_csv(BARPLOT_CSV)
    if len(df) != 6:
        raise SystemExit(f"Expected 6 barplot samples, found {len(df)}")
    return df


def _load_sample(stem: str) -> tuple[np.ndarray, list[str]]:
    ds = ThreeChannelZscoreDataset(
        FRAG_TEST,
        H_TEST,
        L_TEST,
        [FRAG_TEST / f"{stem}.csv"],
    )
    x, _y = ds[0]
    frag = pd.read_csv(FRAG_TEST / f"{stem}.csv")
    win_ids = frag.iloc[:, 0].astype(str).tolist()
    return x.numpy(), win_ids


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


def _integrated_gradients(
    model: Preset29ThreeChannelCNN,
    x: torch.Tensor,
    baseline: torch.Tensor,
    *,
    device: torch.device,
    steps: int = IG_STEPS,
) -> np.ndarray:
    """Return IG attributions with shape (W, 24, 3)."""
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
    """Collapse channel dimension: sum |IG| over 3 input channels -> (W, 24)."""
    return np.abs(ig).sum(axis=-1)


def _chromosome_boundaries(window_ids: list[str]) -> list[tuple[int, str]]:
    marks: list[tuple[int, str]] = []
    prev = None
    for i, wid in enumerate(window_ids):
        chrom = wid.split("_")[0]
        if chrom != prev:
            marks.append((i, chrom.replace("chr", "Chr")))
            prev = chrom
    return marks


def _win_label(wid: str) -> str:
    parts = wid.split("_")
    if len(parts) < 3:
        return wid
    chrom = parts[0].replace("chr", "Chr")
    s = int(parts[1]) // 1_000_000
    e = int(parts[2]) // 1_000_000
    return f"{chrom}:{s}-{e}Mb"


def _plot_heatmap(
    mat: np.ndarray,
    win_ids: list[str],
    *,
    title: str,
    out_png: Path,
    out_pdf: Path,
    dpi: int,
    y_tick_step: int = 40,
    cbar_label: str = "|mean cancer IG − mean healthy IG|",
    label_all_windows: bool = False,
    x_tick_rotation: float = 90,
    x_tick_fontsize: float = 7,
    y_tick_fontsize: float | None = None,
    ylabel: str | None = "Genomic window",
    cbar_tick_fontsize: float = 9,
) -> None:
    n_rows = mat.shape[0]
    if label_all_windows:
        fig_h = max(9.0, n_rows * 0.28)
        fig_w = 12.5
        y_font = y_tick_fontsize if y_tick_fontsize is not None else 6.5
    else:
        fig_h = max(8.0, n_rows * 0.018)
        fig_w = 11.0
        y_font = 7

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Fragment length bin (bp)", fontsize=12, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(np.arange(N_LEN_BINS))
    ax.set_xticklabels(
        BIN_LABELS,
        rotation=x_tick_rotation,
        fontsize=x_tick_fontsize,
        fontweight="bold",
        ha="right" if x_tick_rotation > 0 else "center",
    )

    if label_all_windows:
        y_pos = list(range(n_rows))
    else:
        y_pos = list(range(0, len(win_ids), y_tick_step))
        if len(win_ids) - 1 not in y_pos:
            y_pos.append(len(win_ids) - 1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([_win_label(win_ids[i]) for i in y_pos], fontsize=y_font, fontweight="bold")

    for chrom_i, (xi, _) in enumerate(_chromosome_boundaries(win_ids)):
        if chrom_i > 0:
            ax.axhline(xi - 0.5, color="white", lw=0.4, alpha=0.8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label(cbar_label, fontsize=10, fontweight="bold")
    cbar.ax.tick_params(labelsize=cbar_tick_fontsize, colors="black")
    for t in cbar.ax.get_yticklabels():
        t.set_fontweight("bold")
        t.set_color("black")
    if label_all_windows:
        bottom = 0.16 if x_tick_rotation == 45 else 0.12
        fig.subplots_adjust(left=0.22, right=0.92, top=0.94, bottom=bottom)
    else:
        fig.tight_layout()
    for out in (out_png, out_pdf):
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    targets = _load_targets()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model = _load_model(device)
    tensors: dict[str, np.ndarray] = {}
    win_ids: list[str] | None = None
    for stem in targets["sample_id"]:
        x_np, wids = _load_sample(str(stem))
        tensors[str(stem)] = x_np
        if win_ids is None:
            win_ids = wids
        elif win_ids != wids:
            raise SystemExit(f"window_id mismatch for {stem}")

    assert win_ids is not None
    healthy_stems = targets.loc[targets["group"] == "Healthy", "sample_id"].astype(str).tolist()
    cancer_stems = targets.loc[targets["group"] != "Healthy", "sample_id"].astype(str).tolist()
    baseline = torch.from_numpy(np.mean([tensors[s] for s in healthy_stems], axis=0)).float()

    attrs: dict[str, np.ndarray] = {}
    probs: dict[str, float] = {}
    with torch.no_grad():
        for stem, x_np in tensors.items():
            x = torch.from_numpy(x_np).float()
            probs[stem] = float(torch.sigmoid(model(x.unsqueeze(0).to(device))).item())

    print("Predicted probabilities (rep_031):")
    for stem, p in probs.items():
        print(f"  {stem}: {p:.4f}")

    for stem, x_np in tensors.items():
        x = torch.from_numpy(x_np).float()
        ig = _integrated_gradients(model, x, baseline, device=device, steps=IG_STEPS)
        attrs[stem] = _attr_window_bin(ig)
        print(f"IG done: {stem}")

    cancer_stack = np.stack([attrs[s] for s in cancer_stems], axis=0)
    healthy_stack = np.stack([attrs[s] for s in healthy_stems], axis=0)
    contrast = np.abs(cancer_stack.mean(axis=0) - healthy_stack.mean(axis=0))

    main_title = _ig_main_title(targets)
    stem = "figure5_stage_II_barplot6_mfragnet_ig_window_bin_contrast"
    out_png = args.out_dir / f"{stem}.png"
    out_pdf = args.out_dir / f"{stem}.pdf"
    _plot_heatmap(
        contrast,
        win_ids,
        title=f"{main_title}\nIntegrated Gradients on cancer logit; baseline = mean healthy input",
        out_png=out_png,
        out_pdf=out_pdf,
        dpi=args.dpi,
    )

    top_idx = np.argsort(-contrast.sum(axis=1))[:TOP_ZOOM_WINDOWS]
    top_idx.sort()
    zoom = contrast[top_idx, :]
    zoom_wids = [win_ids[i] for i in top_idx]
    zoom_stem = f"{stem}_top{TOP_ZOOM_WINDOWS}_windows"
    _plot_heatmap(
        zoom,
        zoom_wids,
        title=main_title,
        out_png=args.out_dir / f"{zoom_stem}.png",
        out_pdf=args.out_dir / f"{zoom_stem}.pdf",
        dpi=args.dpi,
        label_all_windows=True,
        x_tick_rotation=45,
        x_tick_fontsize=10,
        ylabel=None,
        y_tick_fontsize=9,
        cbar_tick_fontsize=10,
    )

    rows = []
    for i, wid in enumerate(win_ids):
        rows.append(
            {
                "window_id": wid,
                "total_contrast": float(contrast[i].sum()),
                "max_bin_contrast": float(contrast[i].max()),
                "argmax_bin": BIN_LABELS[int(contrast[i].argmax())],
            }
        )
    rank_df = pd.DataFrame(rows).sort_values("total_contrast", ascending=False)
    rank_csv = args.out_dir / f"{stem}.csv"
    rank_df.to_csv(rank_csv, index=False)
    print(f"Wrote {rank_csv}")
    print(rank_df.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
