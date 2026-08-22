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
│   ├── preprocessing/          # BAM → mFrag-Weight / mFrag-Net feature tables
│   ├── mfrag_weight/           # MFragWeightSL, MFragWeightMDS
│   └── mfrag_net/              # MFragNetFragLength, MFragNetEnd4Motif
├── mfrag_weight/               # trained Lasso models (lung cancer)
├── mfrag_net/                  # trained CNN weights (lung cancer)
├── scripts/
└── examples/
```

## Preprocessing → inference

```python
from mfrag import (
    prepare_fragment_csv,
    build_mfrag_net_inputs,
    build_mfrag_weight_inputs,
    MFragNetFragLength,
    MFragWeightSL,
)

# See code/examples/predict_lung_cancer.py and the root README.md
```

Pre-built feature CSVs can live under `data/lung_cancer/{training,testing}/` or be generated with `build_mfrag_net_inputs` (see `data/README.md`).
