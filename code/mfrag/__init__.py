"""mFrag: methylation-informed cfDNA fragmentomics for lung cancer early detection."""

from mfrag.mfrag_net import MFragNetEnd4Motif, MFragNetFragLength
from mfrag.mfrag_weight import MFragWeightMDS, MFragWeightSL
from mfrag.preprocessing import (
    MfragNetOutputs,
    MfragWeightOutputs,
    annotate_end4_motifs,
    annotate_fragments_gc_weights,
    annotate_reference_methylation,
    build_mfrag_net_inputs,
    build_mfrag_weight_inputs,
    extract_fragments_from_bam,
    prepare_fragment_csv,
)
from mfrag.score import average_proba

__all__ = [
    "MFragWeightSL",
    "MFragWeightMDS",
    "MFragNetFragLength",
    "MFragNetEnd4Motif",
    "average_proba",
    "extract_fragments_from_bam",
    "prepare_fragment_csv",
    "annotate_fragments_gc_weights",
    "annotate_end4_motifs",
    "annotate_reference_methylation",
    "build_mfrag_net_inputs",
    "build_mfrag_weight_inputs",
    "MfragNetOutputs",
    "MfragWeightOutputs",
]

__version__ = "0.1.1"
