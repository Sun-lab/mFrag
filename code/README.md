# mFrag code package

## Install (editable)

```bash
export MFRAG_ROOT=/path/to/mFrag
pip install -e .
```

## Layout

```
code/
├── mfrag/                      # import mfrag
│   ├── coupling_weights.py     # Supplementary File 1 logic
│   ├── paths.py
│   ├── preprocessing/          # BAM pipeline wrappers
│   ├── mfrag_weight/           # MFragWeightSL, MFragWeightMDS
│   └── mfrag_net/              # MFragNetFragLength, MFragNetEnd4Motif
├── mfrag_weight/               # trained Lasso models (lung + ovarian placeholder)
├── mfrag_net/                  # trained CNN weights (lung + ovarian placeholder)
├── scripts/
└── examples/
```

## Inference

```python
from mfrag import MFragNetFragLength
model = MFragNetFragLength.from_pretrained()
p = model.predict_proba("SAMPLE_STEM")
```

Pre-built feature CSVs must live under `data/lung_cancer/{training,testing}/` (see `data/README.md`).
