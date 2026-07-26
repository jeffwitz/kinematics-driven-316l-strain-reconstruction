# First local reconstruction

**Category: Tutorial.** This short path shows what is measured, what is
imposed and what is reconstructed.

## 1. Prepare a small real crop

```bash
source .venv/bin/activate
source /home/jeff/.local/share/tfel/env/env.sh

fem-inhouse prepare-case \
  --raw data/raw/case_study \
  --output data/processed/tutorial-10x10 \
  --crop-nx 10 --crop-ny 10 \
  --hardening-scale-mpa 380 \
  --nonfinite-policy nearest \
  --nodal-completion edge-pad-upper
```

The prepared nodal displacement is measured input. The material maps are local
reconstruction descriptors.

## 2. Solve the local problem

```bash
fem-inhouse partition \
  --input data/processed/tutorial-10x10 \
  --output results/tutorial-local \
  --parts-x 1 --parts-y 1 \
  --padding 0 \
  --increments 10 \
  --constitutive-backend mfront-native-plane-stress \
  --partition-id 0
```

Only boundary displacement is prescribed. Interior displacement, strain,
stress and PEEQ follow from the finite-element equilibrium problem.

## 3. Inspect the distinction

```python
from pathlib import Path
import numpy as np

root = Path("results/tutorial-local")
for path in sorted(root.rglob("*.npy")):
    value = np.load(path, mmap_mode="r")
    print(path.name, value.shape)
```

`U` is a reconstructed nodal field. `E` and `S` are element mechanical fields.
`PEEQ` is an internal plastic variable, not a DIC measurement.

## What you learned

The solver did not smooth DIC strain. It used measured boundary kinematics to
solve for a mechanically admissible interior. Optional complete tensors are
output-only and do not change the 2D solve.

Continue with {doc}`first_coupled_comparison`, or read
{doc}`../explanation/from_dic_to_mechanics`.
