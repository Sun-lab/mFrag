# Preprocessing

Python implementation lives in `code/mfrag/preprocessing/`.

## Typical order

1. `extract_fragments_from_bam` — BAM → fragment CSV
2. `annotate_fragments_gc_weights` — GC-stratum weights (`data/gc_weights/`)
3. `annotate_end4_motifs` — 5′ end 4-mer motifs (requires `pyfaidx` + hg19 FASTA)
4. `annotate_reference_methylation` — healthy/lung WGBS reference β per fragment
5. `build_mfrag_net_inputs` — four 589×24 z-scored matrices for mFrag-Net
6. `build_mfrag_weight_inputs` — row-zscored S/L and MDS wide CSVs for mFrag-Weight

Or use `prepare_fragment_csv` (steps 1–4) followed by steps 5–6.

Coupling weights applied at inference time: `data/supplementary_file_1_coupling_weights.csv`.
