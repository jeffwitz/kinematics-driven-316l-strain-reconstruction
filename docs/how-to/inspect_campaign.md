# Inspect a campaign

**Category: How-to.**

## Check metadata first

Read the campaign manifest and per-partition status before loading large
arrays:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("results/CAMPAIGN/manifest.json")
print(json.dumps(json.loads(p.read_text()), indent=2))
PY
```

Confirm the commit, input hashes, mesh, core and padded bounds, material
backend, solver policy and nonlocal parameters.

## Inspect arrays without copying padded fields

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
root = Path("results/CAMPAIGN/PARTITION")
for name in ("U", "S", "E", "PE", "PEEQ", "RF"):
    a = np.load(root / f"{name}.npy", mmap_mode="r")
    print(name, a.shape, a.dtype, float(np.nanmin(a)), float(np.nanmax(a)))
PY
```

For coupled campaigns also inspect PEEQ nonlocal, mismatch, hardening, yield
radius and coupling residual. Any missing field, incompatible hash or
non-finite value is a failed audit.

## Read diagnostics

Check converged increments, Newton iterations, cutbacks, plane-stress residual,
fixed-point iterations, Helmholtz residual and timing breakdown. Metrics used
for scientific conclusions must be evaluated only on the manifest-defined
core.

Field definitions are in {doc}`../reference/output_contract`; convergence
semantics are in {doc}`../reference/convergence_criteria`.
