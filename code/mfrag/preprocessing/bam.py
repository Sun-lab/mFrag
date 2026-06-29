"""Extract cfDNA fragments from aligned BAM files."""
from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path


def gc_content(seq: str) -> float:
    s = seq.upper().replace("\n", "")
    if not s:
        return 0.0
    return (s.count("G") + s.count("C")) / len(s)


def _load_fai(ref_fasta: str | Path) -> dict:
    fai_path = str(ref_fasta) + ".fai"
    if not os.path.exists(fai_path):
        return {}
    idx: dict = {}
    with open(fai_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 5:
                idx[parts[0]] = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
    return idx


def get_ref_sequence(ref_fasta: str | Path, fai: dict, chrom: str, start_0: int, end_0: int) -> str:
    ref_chrom = chrom if chrom in fai else ("chr" + chrom if "chr" + chrom in fai else "")
    if not ref_chrom:
        return ""
    length, offset, bpl, bpr = fai[ref_chrom]
    if start_0 < 0 or end_0 > length:
        return ""
    seq: list[str] = []
    with open(ref_fasta, "rb") as fa:
        pos = start_0
        while pos < end_0:
            line_i, col = divmod(pos, bpl)
            file_pos = offset + line_i * bpr + col
            to_read = min(bpl - col, end_0 - pos)
            fa.seek(file_pos)
            seq.append(fa.read(to_read).decode("ascii", errors="ignore"))
            pos += to_read
    return "".join(seq)


def extract_fragments_from_bam(
    bam_path: Path | str,
    ref_fasta: Path | str,
    out_csv: Path | str,
    *,
    max_fragments: int | None = None,
) -> int:
    """
    Extract proper-pair fragments from BAM → CSV with columns:
    chromosome, start, end, fragment_length, gc_content.

    Requires ``samtools`` on PATH and hg19 reference FASTA with ``.fai`` index.
    """
    bam_path = str(bam_path)
    ref_fasta = str(ref_fasta)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fai = _load_fai(ref_fasta)
    if not fai:
        raise RuntimeError(f"No .fai index for {ref_fasta}")

    cmd = ["samtools", "view", "-f", "2", "-F", "3844", bam_path]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=65536)

    count = 0
    seen: set[tuple] = set()
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chromosome", "start", "end", "fragment_length", "gc_content"])
        assert proc.stdout is not None
        for line in proc.stdout:
            if max_fragments and count >= max_fragments:
                proc.terminate()
                break
            parts = line.strip().split("\t")
            if len(parts) < 10:
                continue
            qname, flag, chrom, pos_s, _, _, _, _, tlen_s = parts[:9]
            if chrom == "*" or not chrom.startswith("chr"):
                continue
            tlen = abs(int(tlen_s))
            if tlen < 1:
                continue
            pos = int(pos_s) - 1
            key = (qname, pos, tlen)
            if key in seen:
                continue
            seen.add(key)
            end = pos + tlen
            seq = get_ref_sequence(ref_fasta, fai, chrom, pos, end)
            if not seq:
                continue
            writer.writerow([chrom, pos + 1, end, tlen, gc_content(seq)])
            count += 1
    proc.wait()
    return count
