# Preprocessing scripts

Runnable command-line scripts for the mFrag preprocessing pipeline.  
Core logic lives in the importable package `code/mfrag/preprocessing/`.

## Scripts

| Script | Description |
|--------|-------------|
| `extract_fragments_from_bam.py` | BAM → fragment CSV |
| `annotate_fragments.py` | Add GC weights, motifs, reference methylation to an existing fragment CSV |
| `prepare_fragment_csv.py` | BAM → fully annotated fragment CSV (all annotation steps) |
| `build_mfrag_net_inputs.py` | Annotated CSV → four 589×24 z-scored matrices for **mFrag-Net** |
| `build_mfrag_weight_inputs.py` | Annotated CSVs + raw length matrices → row-zscored wide tables for **mFrag-Weight** |

## Example (one sample)

```bash
export MFRAG_ROOT=/path/to/mFrag
export PYTHONPATH=$MFRAG_ROOT/code:$PYTHONPATH

REF=/path/to/hg19.fa
BAM=/path/to/sample.bam
STEM=MY_SAMPLE_lung_cancer
WORK=$MFRAG_ROOT/work/$STEM

# Step 1: BAM → annotated fragment CSV
python code/preprocessing/prepare_fragment_csv.py \
  --bam "$BAM" \
  --ref-fasta "$REF" \
  --out-csv "$WORK/${STEM}.csv" \
  --sample-stem "$STEM" \
  --gc-weights-csv "$MFRAG_ROOT/data/gc_weights/weights_testing.csv"

# Step 2: mFrag-Net inputs
python code/preprocessing/build_mfrag_net_inputs.py \
  --fragments-csv "$WORK/${STEM}.csv" \
  --out-dir "$WORK/mfrag_net"

# Step 3: mFrag-Weight inputs
python code/preprocessing/build_mfrag_weight_inputs.py \
  --fragments-csv "$WORK/${STEM}.csv" \
  --length-raw-dir "$WORK/mfrag_net/fragment_length_raw_weights" \
  --out-dir "$WORK/mfrag_weight"
```

Coupling weights for mFrag-Weight inference: `data/supplementary_file_1_coupling_weights.csv`.
