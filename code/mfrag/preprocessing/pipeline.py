"""High-level preprocessing API for mFrag-Weight and mFrag-Net."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mfrag.preprocessing.bam import extract_fragments_from_bam
from mfrag.preprocessing.gc_weights import annotate_fragments_gc_weights
from mfrag.preprocessing.matrices import (
    build_end4motif_matrices,
    build_fragment_length_matrices,
    build_reference_methylation_matrices,
)
from mfrag.preprocessing.motifs import annotate_end4_motifs
from mfrag.preprocessing.reference_methylation import annotate_reference_methylation
from mfrag.preprocessing.wide_features import (
    build_mfrag_weight_mds_csv,
    build_mfrag_weight_sl_ratio_csv,
)


@dataclass
class MfragNetOutputs:
    """Paths to per-sample z-scored 589×24 matrices for mFrag-Net inference."""

    fragment_length_zscored: Path
    end4motif_zscored: Path
    healthy_ref_methylation_zscored: Path
    lung_ref_methylation_zscored: Path
    sample_stem: str


@dataclass
class MfragWeightOutputs:
    """Paths to row-zscored wide CSVs for mFrag-Weight inference."""

    sl_ratio_row_zscore: Path
    mds_shannon_row_zscore: Path


def prepare_fragment_csv(
    bam_path: Path | str,
    ref_fasta: Path | str,
    out_csv: Path | str,
    *,
    sample_stem: str,
    gc_weights_csv: Path | str,
    healthy_cpg_csv: Path | str,
    lung_cpg_csv: Path | str,
) -> Path:
    """
    Full fragment preprocessing from BAM to model-ready annotated CSV.

    Steps: extract fragments → GC weights → end-4-mer motifs → reference methylation.
    """
    out_csv = Path(out_csv)
    extract_fragments_from_bam(bam_path, ref_fasta, out_csv)
    annotate_fragments_gc_weights(out_csv, gc_weights_csv, sample_stem)
    annotate_end4_motifs(out_csv, ref_fasta)
    annotate_reference_methylation(out_csv, healthy_cpg_csv, lung_cpg_csv)
    return out_csv


def build_mfrag_net_inputs(
    fragments_csv: Path | str,
    out_dir: Path | str,
    *,
    write_raw: bool = True,
) -> MfragNetOutputs:
    """
  Build all four z-scored 589×24 matrix CSVs required by mFrag-Net.

  Input fragment CSV must contain: chromosome, start, end, fragment_length,
  gc_content, weight, end_4_motif_5p, end_4_motif_5m,
  HealthyRef_DNAMethylation, LungcancerRef_DNAMethylation.
    """
    src = Path(fragments_csv)
    out = Path(out_dir)
    stem = src.stem

    frag_z = build_fragment_length_matrices(src, out, write_raw=write_raw, write_row_norm=False)
    motif_z = build_end4motif_matrices(src, out, write_raw=write_raw, write_row_norm=False)
    h_z, l_z = build_reference_methylation_matrices(src, out, write_raw=write_raw, write_row_norm=False)

    return MfragNetOutputs(
        fragment_length_zscored=frag_z,
        end4motif_zscored=motif_z,
        healthy_ref_methylation_zscored=h_z,
        lung_ref_methylation_zscored=l_z,
        sample_stem=stem,
    )


def build_mfrag_weight_inputs(
    fragment_csvs: list[Path | str],
    length_raw_dir: Path | str,
    out_dir: Path | str,
    *,
    reference_training_csvs: list[Path | str] | None = None,
) -> MfragWeightOutputs:
    """
    Build row-zscored wide CSVs for mFrag-Weight (S/L ratio + MDS Shannon).

    ``length_raw_dir`` should contain per-sample ``fragment_length_raw_weights/*.csv``
    (enable ``write_raw=True`` in ``build_mfrag_net_inputs``).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_paths = sorted(Path(length_raw_dir).glob("*.csv"))
    sl_out = build_mfrag_weight_sl_ratio_csv(
        raw_paths,
        out / "slratio_row_zscore.csv",
        reference_training_csvs=reference_training_csvs,
    )
    mds_out = build_mfrag_weight_mds_csv(
        fragment_csvs,
        out / "end4motif_shannon_row_zscore.csv",
        reference_training_csvs=reference_training_csvs,
    )
    return MfragWeightOutputs(sl_ratio_row_zscore=sl_out, mds_shannon_row_zscore=mds_out)
