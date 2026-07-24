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
for name in ("U", "S", "E", "PE", "PEEQ", "RF"):
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

## Interpret convergence

Check at least:

| Check | Expected |
|---|---|
| `converged_increments` | equal to the requested count |
| `cutbacks` | preferably zero; otherwise explained |
| `maximum_newton_iterations` | below the configured limit |
| `final_relative_residual` | below tolerance |
| `backend` | mentions PyPardiso and MFront |
| finite fields | `true` |
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
