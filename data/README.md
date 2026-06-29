# mFrag data directory

All processed study outputs for the LUCAS lung-cancer analysis are hosted in this repository under [Sun-lab/mFrag](https://github.com/Sun-lab/mFrag).

## Layout

```
data/
├── supplementary_file_1_coupling_weights.csv   # per-window S/L & MDS coupling (lung)
├── mfrag_weight/                               # mFrag-Weight input wide tables (row-zscored)
│   ├── slratio_training_row_zscore.csv
│   ├── slratio_testing_row_zscore.csv
│   ├── end4motif_shannon_training_row_zscore.csv
│   └── end4motif_shannon_testing_row_zscore.csv
├── predictions/                                # held-out probabilities (train + test)
│   ├── mfrag_weight_sl_ratio_{training,testing}.csv
│   ├── mfrag_weight_mds_{training,testing}.csv
│   ├── mfrag_net_frag_length_testing.csv
│   └── mfrag_net_end4motif_testing.csv
├── bootstrap/                                  # bootstrap AUC/PR-AUC summaries
│   ├── mfrag_weight_sl_ratio/
│   ├── mfrag_weight_mds/
│   ├── mfrag_net_frag_length/
│   ├── mfrag_net_end4motif/
│   ├── four_model_frag_length/
│   └── four_model_end4motif/
├── external_methylation/
│   └── external_window_mean_beta_lung.csv      # window-level reference Δβ summary
├── gc_weights/
├── cohort/
└── lung_cancer/
    ├── training/   # 287 samples × 4 feature types (589×24 z-scored matrices)
    └── testing/    # 431 samples × 4 feature types
```

## Regenerating outputs

```bash
export MFRAG_ROOT=/path/to/mFrag
export PYTHONPATH=$MFRAG_ROOT/code:$PYTHONPATH
python code/scripts/export_study_results.py
```

Requires a local checkout of the LUCAS feature pipeline (`Mathios/Final_data_24`) for matrix export, or use the committed files directly.

## Supplementary File 1

```bash
python code/scripts/build_supplementary_file_1.py
```

| Column | Description |
|--------|-------------|
| `sl_ratio_coupling_weight` | \|Δβ\| / mean(\|Δβ\|) for mFrag-Weight (S/L) |
| `mds_coupling_weight` | max(1 − λ·Δβ̃, 0.05), λ=0.5 for mFrag-Weight (MDS) |

## External WGBS references (not hosted here)

Pooled single-CpG tables used during preprocessing (~650–775 MB each) are derived from:

- Loyfer *et al.*, *Nature* 613, 355–364 (2023); GEO **GSE186458**
- Zhang *et al.*, *Cell Rep.* 44, 116197 (2025); GEO **GSE212391**

Download processed CpG tables from GEO or rebuild with `code/preprocessing/annotate_fragments.py`.
