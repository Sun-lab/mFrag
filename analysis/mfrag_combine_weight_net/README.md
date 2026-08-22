# Combining mFrag-Weight and mFrag-Net

This folder documents the ensemble used in the manuscript.

## Primary score (manuscript)

For each modality, the reported **mFrag** score is the equal-weight average:

```text
mFrag = 0.5 * (mFrag-Weight + mFrag-Net)
```

- Fragment length: Weight (S/L) + Net (frag length)  
- Motif: Weight (MDS) + Net (end 4-mer)

In code: `from mfrag import average_proba`.

## Learned logistic fusion (supplementary check)

`fit_weight_net_linear_fusion.py` fits

```text
P(y=1) = sigmoid(β0 + β_W * p_Weight + β_N * p_Net)
```

on the **training** set using in-sample base-model probabilities (same protocol as prior Mathios-style ensembles). On the LUCAS testing set, equal averaging (`Mean`) matched or exceeded logistic fusion (`FuseLogit`) for AUC and PR-AUC; see `metrics_summary.csv`. Fitted coefficients were both positive (roughly 1∶2 Weight∶Net for the fragment-length pair; see `fusion_weights.json`).
