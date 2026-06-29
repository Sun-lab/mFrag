"""BAM → mFrag feature preprocessing pipeline (wrappers around Mathios scripts)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from mfrag.paths import MATHIOS_SCRIPTS, MFRAG_ROOT, SUPPLEMENTARY_FILE_1

PIPELINE_STEPS = [
    ("extract_fragments", "extract_fragments_to_csv.py", "BAM → per-sample fragment CSV"),
    ("add_gc_weights", "reorg_fragments_csv_train_test_add_weights.py", "Assign GC stratum weights"),
    ("add_motifs", "add_motif_and_weights_to_fragments.py", "Annotate 5′ end 4-mer motifs"),
    ("add_dnamethylation", "add_dnamethylation_to_fragments_csv.py", "Map external reference β to fragments"),
    ("build_fragment_length", "build_final_data_24_fragment_length.py", "Window × 24-bin fragment length matrices"),
    ("build_end4motif", "build_final_data_24_end4motif.py", "Window × 24-bin end-motif matrices"),
    ("build_dnamethylation", "build_final_data_24_dnamethylation.py", "Window × 24-bin reference methylation matrices"),
    ("build_sl_tables", "build_sl_ratio_wide_tables.py", "Per-sample S/L ratio wide tables"),
    ("build_mds_tables", "build_end4motif_shannon_wide_tables.py", "Per-sample MDS wide tables"),
]


def run_script(script_name: str, *args: str, python: str | None = None) -> None:
    py = python or sys.executable
    script = MATHIOS_SCRIPTS / script_name
    if not script.is_file():
        raise FileNotFoundError(f"Missing preprocessing script: {script}")
    subprocess.run([py, str(script), *args], check=True)


def ensure_coupling_weights() -> Path:
    """Generate Supplementary File 1 if absent."""
    if SUPPLEMENTARY_FILE_1.is_file():
        return SUPPLEMENTARY_FILE_1
    from mfrag.coupling_weights import build_supplementary_file_1
    from mfrag.paths import EXT_METH_LUNG

    build_supplementary_file_1(EXT_METH_LUNG, SUPPLEMENTARY_FILE_1)
    return SUPPLEMENTARY_FILE_1


def list_pipeline_steps() -> list[tuple[str, str, str]]:
    return list(PIPELINE_STEPS)
