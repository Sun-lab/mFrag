# mFrag

**Integrating External Epigenomic References into Cell-Free DNA Fragmentomics for Cancer Early Detection**

mFrag couples publicly available whole-genome bisulfite sequencing (WGBS) methylation references with shallow cfDNA WGS fragmentomics through two complementary models:

- **mFrag-Weight** — explicit Δβ coupling of window-level S/L ratio or MDS features, classified with PCA + Lasso
- **mFrag-Net** — implicit 3-channel CNN integration of fragmentomic profiles and dual reference methylation maps

![mFrag overview](docs/figures/Figure1.png)

## Repository layout

```
mFrag/
├── README.md
├── analysis/                 # Figure reproduction scripts (Fig. 2–6, Sup. Figs. 1–6)
├── code/
│   ├── mfrag/                # Importable Python package
│   │   ├── preprocessing/    # BAM → feature matrix wrappers
│   │   ├── mfrag_weight/     # mFrag-Weight inference
│   │   └── mfrag_net/        # mFrag-Net inference
│   └── scripts/              # CLI utilities
├── data/
│   ├── lung_cancer/          # Per-sample window × 24-bin matrices (symlinks in dev checkout)
│   ├── external_methylation/ # Pooled WGBS references + window Δβ
│   ├── gc_weights/           # GC-correction lookup tables
│   ├── cohort/               # LUCAS train/test metadata
│   └── supplementary_file_1_coupling_weights.csv
└── docs/figures/             # Manuscript figures for README
```

## Quick start

### Environment

```bash
conda create -n mfrag python=3.10
conda activate mfrag
pip install -r requirements.txt
export MFRAG_ROOT=/path/to/mFrag
export PYTHONPATH=$MFRAG_ROOT/code:$PYTHONPATH
```

### Predict lung cancer probability

```python
from mfrag import MFragWeightSL, MFragWeightMDS, MFragNetFragLength, MFragNetEnd4Motif

# mFrag-Weight (requires row-zscored wide CSV or per-sample Series)
sl_model = MFragWeightSL.from_pretrained()
prob_sl = sl_model.predict_proba_from_wide_csv("path/to/slratio_testing_row_zscore.csv")

mds_model = MFragWeightMDS.from_pretrained()
prob_mds = mds_model.predict_proba_from_wide_csv("path/to/end4motif_shannon_testing_row_zscore.csv")

# mFrag-Net (requires pre-built 3-channel z-scored matrices under data/lung_cancer/)
frag_net = MFragNetFragLength.from_pretrained()
prob_frag = frag_net.predict_proba("CGPLLU1P_lung_cancer")

motif_net = MFragNetEnd4Motif.from_pretrained()
prob_motif = motif_net.predict_proba("CGPLLU1P_lung_cancer")
```

See `code/examples/predict_lung_cancer.py` for a runnable example.

### Supplementary File 1 (coupling weights)

Per-window S/L and MDS coupling weights for lung cancer:

```bash
python code/scripts/build_supplementary_file_1.py
# → data/supplementary_file_1_coupling_weights.csv
```

Columns: `window_id`, `healthy_mean_beta`, `lung_cancer_mean_beta`, `abs_delta_beta`, `sl_ratio_coupling_weight`, `mds_db_tilde`, `mds_coupling_weight`.

## Preprocessing pipeline

Input: aligned cfDNA WGS BAMs. Output per sample:

| Output | Description |
|--------|-------------|
| `fragment_length_zscored` | GC-corrected, sample z-scored 589×24-bin length histogram |
| `end4motif_zscored` | GC-corrected, sample z-scored 24 end-4-mer motif bins per window |
| `healthy_ref_methylation_zscored` | External 23-donor cfDNA WGBS reference (24 β-strata bins / window) |
| `lung_ref_methylation_zscored` | External 10 lung-tumor WGBS reference |
| Row-zscored S/L ratio | Window-level DELFI S/L, coupled via Supplementary File 1 |
| Row-zscored MDS | Window-level Shannon entropy, coupled via Supplementary File 1 |

Wrappers live in `code/mfrag/preprocessing/`. Full BAM-to-matrix scripts are in the companion `Mathios/scripts/` pipeline (see `analysis/` README files).

## Trained models (lung cancer)

| Model | Path |
|-------|------|
| mFrag-Weight (S/L) | `code/mfrag_weight/mfrag_weight_sl_ratio/lung_cancer/models/` |
| mFrag-Weight (MDS) | `code/mfrag_weight/mfrag_weight_mds/lung_cancer/models/` |
| mFrag-Net (Frag length, rep_031) | `code/mfrag_net/mfrag_net_frag_length/lung_cancer/models/` |
| mFrag-Net (End 4-mer motif, rep_006) | `code/mfrag_net/mfrag_net_end4motif/lung_cancer/models/` |

Ovarian cancer model slots are reserved under `*/ovarian_cancer/` (not yet populated).

## Reproducing figures

| Manuscript panel | Scripts directory |
|------------------|-------------------|
| Figure 2 | `analysis/figure2/` |
| Figure 3 | `analysis/figure3/` |
| Figure 4 | `analysis/figure4/` |
| Figure 5 | `analysis/figure5/` |
| Figure 6 | `analysis/figure6/` |
| Supplementary Fig. 1 | `analysis/supplementary_figure1/` |
| Supplementary Fig. 2 | `analysis/supplementary_figure2/` |
| Supplementary Figs. 3–6 | `analysis/supplementary_figure3/` … `supplementary_figure6/` |

Compiled figures: `../mFrag_Figures/Fianl_Figures/`.

## External validation (prostate)

![Prostate external validation](docs/figures/FigureS7.png)

Supplementary Figure 7 scripts: `../mFrag_Figures/Figure S7/` (symlinked from `analysis/`).

## Citation

```
Cui S, …, Sun W. mFrag: Integrating External Epigenomic References into Cell-Free DNA
Fragmentomics for Cancer Early Detection. Nature Methods (in preparation).
```

## License

[Specify license, e.g. MIT]

## Contact

Wei Sun — [email]
