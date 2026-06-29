"""Shared Figure 4 plot titles (ROC / PR / bootstrap boxplots)."""
from __future__ import annotations

import pandas as pd


def bootstrap_plot_title(
    split_key: str,
    split_label: str,
    sample_ids: list[str],
    pool_summary: pd.DataFrame | None = None,
) -> str:
    if split_key == "overall":
        n_healthy = sum(1 for s in sample_ids if str(s).endswith("_healthy"))
        n_cancer = sum(1 for s in sample_ids if str(s).endswith("_lung_cancer"))
        return f"All stages (testing healthy = {n_healthy}, testing cancer = {n_cancer})"

    if pool_summary is None:
        raise ValueError("pool_summary required for stage splits")
    row = pool_summary.loc[pool_summary["stage_group"] == split_key].iloc[0]
    n_healthy = int(row["n_healthy"])
    n_cancer = int(row["n_cancer_in_stage"])
    return f"{split_label} (testing healthy = {n_healthy}, testing cancer = {n_cancer})"
