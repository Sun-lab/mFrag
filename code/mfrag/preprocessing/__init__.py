"""mFrag preprocessing: BAM → model-ready feature tables."""
from __future__ import annotations

from mfrag.preprocessing.bam import extract_fragments_from_bam
from mfrag.preprocessing.gc_weights import annotate_fragments_gc_weights
from mfrag.preprocessing.matrices import (
    build_end4motif_matrices,
    build_fragment_length_matrices,
    build_reference_methylation_matrices,
)
from mfrag.preprocessing.motifs import annotate_end4_motifs
from mfrag.preprocessing.pipeline import (
    MfragNetOutputs,
    MfragWeightOutputs,
    build_mfrag_net_inputs,
    build_mfrag_weight_inputs,
    prepare_fragment_csv,
)
from mfrag.preprocessing.reference_methylation import (
    annotate_reference_methylation,
    load_pooled_cpg_index,
)
from mfrag.preprocessing.wide_features import (
    build_mfrag_weight_mds_csv,
    build_mfrag_weight_sl_ratio_csv,
    row_zscore_wide,
)

__all__ = [
    "extract_fragments_from_bam",
    "annotate_fragments_gc_weights",
    "annotate_end4_motifs",
    "annotate_reference_methylation",
    "load_pooled_cpg_index",
    "build_fragment_length_matrices",
    "build_end4motif_matrices",
    "build_reference_methylation_matrices",
    "build_mfrag_net_inputs",
    "build_mfrag_weight_inputs",
    "build_mfrag_weight_sl_ratio_csv",
    "build_mfrag_weight_mds_csv",
    "row_zscore_wide",
    "prepare_fragment_csv",
    "MfragNetOutputs",
    "MfragWeightOutputs",
]
