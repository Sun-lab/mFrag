# mFrag

**Integrating External Epigenomic References into Cell-Free DNA Fragmentomics for Cancer Early Detection**

mFrag couples publicly available whole-genome bisulfite sequencing (WGBS) methylation references with shallow cfDNA WGS fragmentomics through two complementary models:

- **mFrag-Weight** — explicit Δβ coupling of window-level S/L ratio or MDS (Shannon entropy) features, classified with PCA + Lasso
- **mFrag-Net** — implicit 3-channel CNN integration of fragmentomic profiles and dual reference methylation maps

## Repository layout

```
mFrag/
├── README.md
├── requirements.txt
├── code/
│   ├── mfrag/                    # Importable Python package
│   │   ├── preprocessing/        # preprocessing API (import mfrag)
│   │   ├── mfrag_weight/         # mFrag-Weight inference
│   │   └── mfrag_net/            # mFrag-Net inference
│   ├── preprocessing/            # CLI scripts (run from shell)
│   ├── mfrag_weight/.../models/  # Pretrained lung-cancer checkpoints
│   ├── mfrag_net/.../models/
│   ├── scripts/                  # Other CLI utilities
│   └── examples/                 # Runnable demos
└── data/
    ├── external_methylation/     # Pooled WGBS reference tables
    ├── gc_weights/               # GC-correction lookup tables
    ├── cohort/                   # Example cohort metadata
    └── supplementary_file_1_coupling_weights.csv
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

Coupling weights for mFrag-Weight are in `data/supplementary_file_1_coupling_weights.csv` (also rebuildable via `code/scripts/build_supplementary_file_1.py`).

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

# BAM → annotated fragment table (GC weights, motifs, reference methylation)
prepare_fragment_csv(
    bam_path="/path/to/sample.bam",
    ref_fasta="/path/to/hg19.fa",
    out_csv=frag_csv,
    sample_stem=sample,
    gc_weights_csv=WEIGHTS_TESTING,
    healthy_cpg_csv=POOLED_HEALTHY_CPG,
    lung_cpg_csv=POOLED_LUNG_CPG,
)

# mFrag-Net: four 589×24 z-scored matrix CSVs
net_inputs = build_mfrag_net_inputs(frag_csv, out / "mfrag_net", write_raw=True)

# mFrag-Weight: row-zscored wide CSVs (S/L + MDS)
weight_inputs = build_mfrag_weight_inputs(
    fragment_csvs=[frag_csv],
    length_raw_dir=out / "mfrag_net" / "fragment_length_raw_weights",
    out_dir=out / "mfrag_weight",
)
```

If you already have an annotated fragment CSV, skip `prepare_fragment_csv` and call `build_mfrag_net_inputs` / `build_mfrag_weight_inputs` directly.

### 2. Predict lung-cancer probability

```python
from mfrag import (
    MFragWeightSL,
    MFragWeightMDS,
    MFragNetFragLength,
    MFragNetEnd4Motif,
)

sample = "MY_SAMPLE_lung_cancer"

# mFrag-Weight (row-zscored wide CSVs from step 1)
sl_model = MFragWeightSL.from_pretrained()
mds_model = MFragWeightMDS.from_pretrained()
prob_sl, names = sl_model.predict_proba_from_wide_csv(weight_inputs.sl_ratio_row_zscore)
prob_mds, _ = mds_model.predict_proba_from_wide_csv(weight_inputs.mds_shannon_row_zscore)

# mFrag-Net (z-scored 589×24 matrices from step 1)
frag_net = MFragNetFragLength.from_pretrained(
    main_dir=net_inputs.fragment_length_zscored.parent,
    healthy_ref_dir=net_inputs.healthy_ref_methylation_zscored.parent,
    lung_ref_dir=net_inputs.lung_ref_methylation_zscored.parent,
)
motif_net = MFragNetEnd4Motif.from_pretrained(
    main_dir=net_inputs.end4motif_zscored.parent,
    healthy_ref_dir=net_inputs.healthy_ref_methylation_zscored.parent,
    lung_ref_dir=net_inputs.lung_ref_methylation_zscored.parent,
)

print("mFrag-Weight S/L:     ", prob_sl[names.index(sample)])
print("mFrag-Weight MDS:     ", prob_mds[names.index(sample)])
print("mFrag-Net frag length:", frag_net.predict_proba(sample))
print("mFrag-Net end 4-mer:  ", motif_net.predict_proba(sample))
```

See `code/examples/predict_lung_cancer.py` for a runnable example using bundled reference tables.

## Preprocessing API reference

| Function | Description |
|----------|-------------|
| `extract_fragments_from_bam` | BAM → fragment CSV |
| `annotate_fragments_gc_weights` | Add GC-stratum `weight` column |
| `annotate_end4_motifs` | Add `end_4_motif_5p` / `end_4_motif_5m` |
| `annotate_reference_methylation` | Add healthy/lung WGBS reference β |
| `prepare_fragment_csv` | All of the above in one call |
| `build_mfrag_net_inputs` | Annotated CSV → four z-scored 589×24 matrices |
| `build_mfrag_weight_inputs` | Annotated CSV + raw length matrices → two wide tables |
| `build_mfrag_weight_sl_ratio_csv` | S/L ratio wide table only |
| `build_mfrag_weight_mds_csv` | MDS Shannon wide table only |

## Pretrained models (lung cancer)

| Model | Location |
|-------|----------|
| mFrag-Weight (S/L) | `code/mfrag_weight/mfrag_weight_sl_ratio/lung_cancer/models/` |
| mFrag-Weight (MDS) | `code/mfrag_weight/mfrag_weight_mds/lung_cancer/models/` |
| mFrag-Net (fragment length) | `code/mfrag_net/mfrag_net_frag_length/lung_cancer/models/` |
| mFrag-Net (end 4-mer motif) | `code/mfrag_net/mfrag_net_end4motif/lung_cancer/models/` |

Ovarian cancer model slots are reserved under `*/ovarian_cancer/` (not yet populated).

## Data requirements

- **Pooled WGBS CpG tables** (~650–775 MB each): required for `annotate_reference_methylation`. Download instructions in `data/README.md`.
- **Per-sample feature matrices** for mFrag-Net inference: generated by `build_mfrag_net_inputs` or provided under `data/lung_cancer/` in a full checkout.
- **mFrag-Weight** requires scikit-learn ≥ 1.7 (models trained with 1.7.2).

## Citation

```
Cui S, …, Sun W. mFrag: Integrating External Epigenomic References into Cell-Free DNA
Fragmentomics for Cancer Early Detection. Nature Methods (in preparation).
```

## License

[Specify license, e.g. MIT]

## Contact

Wei Sun — [email]
