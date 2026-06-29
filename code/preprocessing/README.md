# Preprocessing

BAM → mFrag feature matrices.

Implementation: `code/mfrag/preprocessing/` (Python API) wrapping `Mathios/scripts/` pipeline steps.

Typical order:
1. `extract_fragments_to_csv.py`
2. `reorg_fragments_csv_train_test_add_weights.py`
3. `add_motif_and_weights_to_fragments.py`
4. `add_dnamethylation_to_fragments_csv.py`
5. `build_final_data_24_fragment_length.py`
6. `build_final_data_24_end4motif.py`
7. `build_final_data_24_dnamethylation.py`
8. `build_sl_ratio_wide_tables.py` + `row_zscore_slratio_tables.py`
9. `build_end4motif_shannon_wide_tables.py` + `row_zscore_end4motif_shannon_tables.py`

Coupling weights: `data/supplementary_file_1_coupling_weights.csv`
