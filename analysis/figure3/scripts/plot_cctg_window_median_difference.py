#!/usr/bin/env python3
"""Backward-compatible entry point for CCTG window median-difference profile."""
from plot_motif_window_median_difference import main

if __name__ == "__main__":
    import sys

    if "--motifs" not in sys.argv:
        sys.argv[1:1] = ["--motifs", "CCTG"]
    main()
