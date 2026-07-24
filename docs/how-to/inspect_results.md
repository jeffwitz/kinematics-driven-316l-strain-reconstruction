# Inspect and validate a campaign

This guide audits an existing campaign without rerunning the solve.

## Check a partition status

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

status_path = Path(
    "validation/reference_data/"
    "article_100p_pad150_p0000_mfront_v1/"
    "partitions/0000/status.json"
)
status = json.loads(status_path.read_text())
print("complete:", status["complete"])
print("backend:", status["diagnostics"]["backend"])
print("increments:", status["diagnostics"]["converged_increments"])
print("cutbacks:", status["diagnostics"]["cutbacks"])
print("relative residual:", status["diagnostics"]["final_relative_residual"])
PY
```

A complete flag is not enough: every field fingerprint must match as well.

## Regenerate the article-partition report

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/validate_saved_article_partition.py \
  --campaign \
    validation/reference_data/article_100p_pad150_p0000_mfront_v1 \
  --input data/processed/case_study \
  --partition-id 0 \
  --comparison-campaign \
    validation/reference_data/article_100p_pad150_p0000
```

The command:

- checks shapes, types, finite values, and SHA-256 fingerprints;
- checks prescribed DIC displacement on the boundary;
- checks global reaction balance;
- computes von Mises stress;
- derives DIC and FE equivalent strain with the same operator;
- compares MFront with the saved tabulated Python campaign;
- writes derived maps and `validation-report.json`.

It never modifies the six raw result fields.

## Check all arrays quickly

```bash
.venv/bin/python - <<'PY'
from pathlib import Path

import numpy as np

partition = Path("results/reconstruction-100/partitions/0000")
for name in (
    "U", "S", "E", "PE", "PEEQ", "RF",
    "S_3D", "E_3D", "EE_3D", "PE_3D", "S33_RESIDUAL_MPA",
):
    field = np.load(partition / f"{name}.npy", mmap_mode="r")
    print(
        f"{name:4s}",
        "shape=", field.shape,
        "dtype=", field.dtype,
        "finite=", bool(np.isfinite(field).all()),
        "min=", float(np.min(field)),
        "max=", float(np.max(field)),
    )
PY
```

## Check the complete plane-stress state

```bash
.venv/bin/python - <<'PY'
from pathlib import Path

import numpy as np

from fem_inhouse import load_full_tensor_state

partition = Path("results/reconstruction-100/partitions/0000")
state = load_full_tensor_state(partition)

additive = (
    state.total_strain_tensor
    - state.elastic_strain_tensor
    - state.plastic_strain_tensor
)
plastic_trace = np.trace(state.plastic_strain_tensor, axis1=-2, axis2=-1)

print("maximum |S33| (MPa):", np.max(np.abs(state.plane_stress_residual_mpa)))
print("maximum |trace(PE)|:", np.max(np.abs(plastic_trace)))
print("maximum |E - EE - PE|:", np.max(np.abs(additive)))
PY
```

For an older directory containing only `S.npy`, `E.npy`, and `PE.npy`, call
`load_full_tensor_state(partition, poisson_ratio=0.3)`. The material property
is mandatory because it cannot be inferred from those arrays.

## Interpret convergence

Check at least:

| Check | Expected |
|---|---|
| `converged_increments` | equal to the requested count |
| `cutbacks` | preferably zero; otherwise explained |
| `maximum_newton_iterations` | below the configured limit |
| `final_relative_residual` | below tolerance |
| `backend` | mentions PyPardiso and MFront |
| `tensor_reconstruction_source` | `mfront_native_axial_strain` for the nominal backend |
| finite fields | `true` |
| `S_3D[..., 2, 2]` | exactly equal to `S33_RESIDUAL_MPA` |
| plastic trace and additive residual | below the declared tensor tolerances |
| total reaction resultant | close to zero |
| DIC boundary error | close to machine precision |

A converged residual does not prove scientific validity. It proves only that
the configured discrete problem was solved.

## Inspect maps

The saved article campaign contains:

```text
validation/reference_data/
└── article_100p_pad150_p0000_mfront_v1/
    ├── preview.png
    ├── derived/
    │   ├── DIC_EVM.npy
    │   ├── FEM_EVM.npy
    │   ├── DIFF_EVM.npy
    │   └── S_MISES.npy
    └── validation-report.json
```

Look for:

- continuity of the principal localization bands;
- an artificial grid at core interfaces;
- isolated extrema associated with material maps;
- consistency between plastic zones and PEEQ;
- bias and spatial correlation, not only RMSE and MAE.

{doc}`../explanation/validation` explains what these checks can—and cannot—
establish.
