"""Repository path helpers."""
from __future__ import annotations

import os
from pathlib import Path

MFRAG_ROOT = Path(os.environ.get("MFRAG_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = MFRAG_ROOT / "data"
CODE_DIR = MFRAG_ROOT / "code"

EXT_METH_LUNG = DATA_DIR / "external_methylation" / "external_window_mean_beta_lung.csv"
SUPPLEMENTARY_FILE_1 = DATA_DIR / "supplementary_file_1_coupling_weights.csv"
GC_WEIGHTS_REF = DATA_DIR / "gc_weights" / "fragments_by_strata_reference.csv"

LUNG_TRAIN_FRAG = DATA_DIR / "lung_cancer" / "training" / "fragment_length_zscored"
LUNG_TEST_FRAG = DATA_DIR / "lung_cancer" / "testing" / "fragment_length_zscored"
LUNG_TRAIN_MOTIF = DATA_DIR / "lung_cancer" / "training" / "end4motif_zscored"
LUNG_TEST_MOTIF = DATA_DIR / "lung_cancer" / "testing" / "end4motif_zscored"
LUNG_TRAIN_HREF = DATA_DIR / "lung_cancer" / "training" / "healthy_ref_methylation_zscored"
LUNG_TRAIN_LREF = DATA_DIR / "lung_cancer" / "training" / "lung_ref_methylation_zscored"
LUNG_TEST_HREF = DATA_DIR / "lung_cancer" / "testing" / "healthy_ref_methylation_zscored"
LUNG_TEST_LREF = DATA_DIR / "lung_cancer" / "testing" / "lung_ref_methylation_zscored"

MFRAG_WEIGHT_SL_LUNG = CODE_DIR / "mfrag_weight" / "mfrag_weight_sl_ratio" / "lung_cancer" / "models"
MFRAG_WEIGHT_MDS_LUNG = CODE_DIR / "mfrag_weight" / "mfrag_weight_mds" / "lung_cancer" / "models"
MFRAG_NET_FRAG_LUNG = CODE_DIR / "mfrag_net" / "mfrag_net_frag_length" / "lung_cancer" / "models"
MFRAG_NET_MOTIF_LUNG = CODE_DIR / "mfrag_net" / "mfrag_net_end4motif" / "lung_cancer" / "models"

# Bundled reference data defaults
POOLED_HEALTHY_CPG = DATA_DIR / "external_methylation" / "pooled_23samples_cfDNA_single_cpg_sites.csv"
POOLED_LUNG_CPG = DATA_DIR / "external_methylation" / "pooled_10samples_lung_single_cpg_sites.csv"
WEIGHTS_TRAINING = DATA_DIR / "gc_weights" / "weights_training.csv"
WEIGHTS_TESTING = DATA_DIR / "gc_weights" / "weights_testing.csv"
