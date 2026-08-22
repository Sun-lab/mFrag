# mFrag

**Integrating External Epigenomic References into Cell-Free DNA Fragmentomics for Cancer Early Detection**

<p align="center">
  <img src="docs/figures/Figure1.png" alt="mFrag overview (Figure 1)" width="90%"/>
</p>

<p align="center"><em>Figure 1.</em> Overview of methylation-informed fragmentomics. Shallow cfDNA WGS yields window-level fragment-length and end-motif features; external WGBS references (healthy cfDNA and lung tumor) inform <strong>mFrag-Weight</strong> (explicit Δβ coupling + PCA/Lasso) and <strong>mFrag-Net</strong> (3-channel CNN). The primary <strong>mFrag</strong> score is the equal-weight average of the two model probabilities for the matched modality.</p>

mFrag couples publicly available whole-genome bisulfite sequencing (WGBS) methylation references with shallow cfDNA WGS fragmentomics through two complementary models:

- **mFrag-Weight** — explicit Δβ coupling of window-level S/L ratio or MDS (Shannon entropy) features, classified with PCA + Lasso
- **mFrag-Net** — implicit 3-channel CNN integration of fragmentomic profiles and dual reference methylation maps
- **mFrag score (primary)** — for each modality,  
  `mFrag = 0.5 × (mFrag-Weight + mFrag-Net)`  
  - Fragment length: Weight (S/L) + Net (frag length)  
  - Motif: Weight (MDS) + Net (end 4-mer)

A learned logistic fusion of the two probabilities was also evaluated; equal averaging performed as well or better on held-out testing AUC/PR-AUC and is used in the manuscript.

## Repository layout

```
mFrag/
├── README.md
├── requirements.txt
├── code/
│   ├── mfrag/                    # Importable Python package
│   │   ├── preprocessing/        # BAM → model inputs
│   │   ├── mfrag_weight/         # mFrag-Weight inference
│   │   ├── mfrag_net/            # mFrag-Net inference
│   │   └── score.py              # average_proba → primary mFrag score
│   ├── preprocessing/            # CLI scripts
│   ├── mfrag_weight/.../models/  # Pretrained lung-cancer checkpoints
│   ├── mfrag_net/.../models/
│   └── examples/                 # Runnable demos
├── data/
│   ├── predictions/              # Per-model and averaged mFrag scores
│   ├── mfrag_weight/             # Row-zscored wide feature tables
│   ├── bootstrap/                # Bootstrap AUC / PR-AUC summaries
│   └── ...
├── analysis/                     # Figure / manuscript analysis scripts
├── docs/
│   ├── figures/Figure1.png
│   ├── DATA_AVAILABILITY.md
│   └── STATS.md                  # Paired DeLong / bootstrap testing notes
└── scripts/
```

## Installation

```bash
conda create -n mfrag python=3.10
conda activate mfrag
pip install -r requirements.txt

export MFRAG_ROOT=/path/to/mFrag
export PYTHONPATH=$MFRAG_ROOT/code:$PYTHONPATH
```

**External tools:** `samtools` (BAM extraction), hg19 reference FASTA with `.fai` index.

## Workflow overview

| Stage | Output | Used by |
|-------|--------|---------|
| Fragment CSV from BAM | `chromosome,start,end,fragment_length,gc_content` | all models |
| + GC weights + motifs + ref. methylation | annotated fragment CSV | all models |
| 589×24 z-scored matrices | fragment length, end-4-mer, healthy/lung ref. β | **mFrag-Net** |
| Row-zscored wide tables | S/L ratio, MDS Shannon entropy | **mFrag-Weight** |
| Probabilities → average | primary **mFrag** score | manuscript reporting |

Coupling weights for mFrag-Weight are in `data/supplementary_file_1_coupling_weights.csv` (also rebuildable via `code/scripts/build_supplementary_file_1.py`).

**Processed study data** (matrices, predictions, bootstrap summaries) are hosted in [`data/`](data/); see [`data/README.md`](data/README.md) and [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md).

## Quick start

### 1. Preprocess one sample (BAM → model inputs)

```python
from pathlib import Path
from mfrag import (
    prepare_fragment_csv,
    build_mfrag_net_inputs,
    build_mfrag_weight_inputs,
)
from mfrag.paths import (
    MFRAG_ROOT,
    POOLED_HEALTHY_CPG,
    POOLED_LUNG_CPG,
    WEIGHTS_TESTING,
)

sample = "MY_SAMPLE_lung_cancer"
out = MFRAG_ROOT / "work" / sample
frag_csv = out / f"{sample}.csv"

prepare_fragment_csv(
    bam_path="/path/to/sample.bam",
    ref_fasta="/path/to/hg19.fa",
    out_csv=frag_csv,
    sample_stem=sample,
    gc_weights_csv=WEIGHTS_TESTING,
    healthy_cpg_csv=POOLED_HEALTHY_CPG,
    lung_cpg_csv=POOLED_LUNG_CPG,
)

net_inputs = build_mfrag_net_inputs(frag_csv, out / "mfrag_net", write_raw=True)
weight_inputs = build_mfrag_weight_inputs(
    fragment_csvs=[frag_csv],
    length_raw_dir=out / "mfrag_net" / "fragment_length_raw_weights",
    out_dir=out / "mfrag_weight",
)
```

### 2. Predict and form the primary mFrag score

```python
from mfrag import (
    MFragWeightSL,
    MFragWeightMDS,
    MFragNetFragLength,
    MFragNetEnd4Motif,
    average_proba,
)

sample = "MY_SAMPLE_lung_cancer"

# --- Fragment-length modality ---
sl_model = MFragWeightSL.from_pretrained()
frag_net = MFragNetFragLength.from_pretrained(
    main_dir=net_inputs.fragment_length_zscored.parent,
    healthy_ref_dir=net_inputs.healthy_ref_methylation_zscored.parent,
    lung_ref_dir=net_inputs.lung_ref_methylation_zscored.parent,
)
prob_sl, names = sl_model.predict_proba_from_wide_csv(weight_inputs.sl_ratio_row_zscore)
p_weight = float(prob_sl[names.index(sample)])
p_net = float(frag_net.predict_proba(sample))
mfrag_frag = average_proba(p_weight, p_net)  # primary mFrag (Frag length)

# --- Motif modality (optional) ---
mds_model = MFragWeightMDS.from_pretrained()
motif_net = MFragNetEnd4Motif.from_pretrained(
    main_dir=net_inputs.end4motif_zscored.parent,
    healthy_ref_dir=net_inputs.healthy_ref_methylation_zscored.parent,
    lung_ref_dir=net_inputs.lung_ref_methylation_zscored.parent,
)
prob_mds, _ = mds_model.predict_proba_from_wide_csv(weight_inputs.mds_shannon_row_zscore)
mfrag_motif = average_proba(
    float(prob_mds[names.index(sample)]),
    float(motif_net.predict_proba(sample)),
)

print("mFrag-Weight S/L:", p_weight)
print("mFrag-Net frag:  ", p_net)
print("mFrag (Frag length):", mfrag_frag)
print("mFrag (Motif):      ", mfrag_motif)
```

See `code/examples/predict_lung_cancer.py` for a runnable demo.

## Pretrained models (lung cancer)

| Model | Location |
|-------|----------|
| mFrag-Weight (S/L) | `code/mfrag_weight/mfrag_weight_sl_ratio/lung_cancer/models/` |
| mFrag-Weight (MDS) | `code/mfrag_weight/mfrag_weight_mds/lung_cancer/models/` |
| mFrag-Net (fragment length) | `code/mfrag_net/mfrag_net_frag_length/lung_cancer/models/` |
| mFrag-Net (end 4-mer motif) | `code/mfrag_net/mfrag_net_end4motif/lung_cancer/models/` |

## Study outputs in `data/predictions/`

| File | Description |
|------|-------------|
| `mfrag_weight_sl_ratio_{training,testing}.csv` | Weight (S/L) probabilities |
| `mfrag_net_frag_length_{training,testing}.csv` | Net (frag length) probabilities |
| `mfrag_frag_length_{training,testing}.csv` | **Primary mFrag (Frag length)** = average |
| `mfrag_weight_mds_*.csv` / `mfrag_net_end4motif_*.csv` | Motif-modality components |
| `mfrag_motif_testing.csv` | **Primary mFrag (Motif)** = average |

## Statistical comparisons (manuscript)

Primary pairwise model contrasts use **paired** tests on the same testing individuals: DeLong for AUC plus paired bootstrap ΔAUC / ΔPR-AUC (PR-AUC: bootstrap only), with Holm adjustment within each panel. See [`docs/STATS.md`](docs/STATS.md).

## Data requirements

- **Pooled WGBS CpG tables** (~650–775 MB each): required for `annotate_reference_methylation`. Download instructions in `data/README.md`.
- **mFrag-Weight** requires scikit-learn ≥ 1.7 (models trained with 1.7.2).

## Citation

```
Cui S, …, Sun W. mFrag: Integrating External Epigenomic References into Cell-Free DNA
Fragmentomics for Cancer Early Detection. (in preparation).
```

## License

[Specify license, e.g. MIT]

## Contact

Wei Sun — Fred Hutchinson Cancer Center / University of Washington
