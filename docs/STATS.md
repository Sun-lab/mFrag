# Statistical comparisons in the mFrag manuscript

Unless noted otherwise, pairwise contrasts of model performance use the **same testing individuals** for both models (paired design).

## Tests

| Metric | Primary tests |
|--------|----------------|
| **AUC** | DeLong test on full testing scores **and** paired bootstrap test of ΔAUC (shared resample indices) |
| **PR-AUC** | Paired bootstrap test of ΔPR-AUC only (DeLong not applicable) |

- Bootstrap: typically **1,000** nonparametric resamples of testing individuals with replacement.
- Multiplicity: **Holm** adjustment within each metric and analysis panel.
- Figure annotations for AUC use the **more conservative** of the Holmed DeLong and Holmed bootstrap *P* values; PR-AUC annotations use the Holmed bootstrap *P* value.
- Alternatives match the scientific contrast (e.g. one-sided favoring mFrag over DELFI in primary figures; two-sided for selected supplementary ablations).

Independent Mann–Whitney tests on bootstrap AUC clouds are **not** used for primary inference (they treat dependent bootstrap replicates as independent samples).

## Ensemble score

The primary **mFrag** probability is the equal-weight average of mFrag-Weight and mFrag-Net for the matched modality (`mfrag.average_proba`). A logistic fusion of the two probabilities was evaluated on the training set; equal averaging performed as well or better on held-out testing AUC/PR-AUC (see `analysis/mfrag_combine_weight_net/`).
