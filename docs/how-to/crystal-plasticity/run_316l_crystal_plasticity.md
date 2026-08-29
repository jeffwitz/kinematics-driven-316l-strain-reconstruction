# Run 316L crystal plasticity

**Mode:** how-to  
**Domain:** crystal-plasticity

## Goal

Run the registered P43 crystal-plasticity case with the twelve-system SRIX law
under a three-traction structural plane-stress closure. The command below uses
the registered 20x20 M20 crop and eight increments, with the homogeneous
orientation control; an EBSD HDF5 source can be supplied when that payload is
available.

## Prerequisites

Build the MFront behaviour and expose it to the driver:

```bash
source /home/jeff/.local/share/tfel/env/env.sh
./scripts/build_mfront_behaviour.sh
export MFRONT_BEHAVIOUR_LIBRARY="$PWD/build/mfront/src/libBehaviour.so"
```

The registered P43 displacement/material maps are loaded from
`data/processed/case_study`. The paired parameter set is
`316l_guilhem2013_nasri2018_meric_srix_rate_1e-3`. Verify the Bunge orientation
convention and the case provenance in
{doc}`../../reference/scientific/ebsd_orientation_contract`.

## Run

Use the structural MFront backend as the recommended reference for a crystal
law:

```bash
MFRONT_BEHAVIOUR_LIBRARY="$PWD/build/mfront/src/libBehaviour.so" \
PYTHONPATH=src python scripts/qualify_crystal_tet2_p43.py \
  --behaviour fcc_forest_rubin_srix \
  --material-backend mfront-structural-plane-stress \
  --crop-nodes 1610 1630 1075 1095 \
  --paired-parameter-set \
    316l_guilhem2013_nasri2018_meric_srix_rate_1e-3 \
  --increments 8 --tolerance 1e-8 --mfront-threads 4 \
  --gps-composite-fd-tangent \
  --output validation/_generated/performance/srix_p43_m20_structural.json
```

Without `--ebsd-orientation-h5`, the driver records its declared homogeneous
Bunge orientation `(35, 20, 15)` in the report. To use a co-registered EBSD
payload, add:

```bash
--ebsd-orientation-h5 path/to/CP_dataset.h5
```

The same law can be run through the independent external closure by changing
`--material-backend` to `mfront-3d-condensed-plane-stress`. This changes where
the three-traction closure is solved, not the SRIX law.

## Expected outputs

For an output stem `srix_p43_m20_structural.json`, the driver writes:

* the JSON report with parameters, orientation, backend, residuals and solver
  diagnostics;
* `srix_p43_m20_structural.fields.npz` with displacement, stress and slip
  observables;
* `srix_p43_m20_structural.progress.jsonl` with the incremental trace.

The field archive contains `stress_in_plane_mpa`, `accumulated_slip`, and,
when exposed by the behaviour, signed and equivalent slip arrays. Units are
millimetres and MPa.

## Verify

```bash
jq '{status, mesh, increments, behaviour, mfront_threads, final_residual,
     newton_iterations, krylov_iterations, orientation, field_file}' \
  validation/_generated/performance/srix_p43_m20_structural.json

python3 - <<'PY'
import numpy as np
path = "validation/_generated/performance/srix_p43_m20_structural.fields.npz"
with np.load(path) as fields:
    print(sorted(fields.files))
    for name in fields.files:
        print(name, fields[name].shape, fields[name].dtype)
PY
```

Check the final residual and the reported plane-stress residual before reading
slip maps. The structural closure targets
`sigma_zz = sigma_xz = sigma_yz = 0` at every local update.

## What this establishes

It demonstrates that the registered 316L SRIX law, orientation convention,
MFront implementation and structural plane-stress driver can execute together
on the declared case.

## What this does **not** establish

The registered parameter set is not an experimental identification of 316L;
the current `R` value is the documented analytical transposition. A successful
run is execution/qualification evidence, not proof that SRIX reproduces every
experimental field. Use {doc}`qualify_native_srix_backend` for the native
implementation comparison.

## See also

* {doc}`../../explanation/constitutive/forest_rubin_srix`
* {doc}`../../reference/scientific/srix_parameter_sets`
* {doc}`../../reference/numerics/plane_stress`
* {doc}`../../reference/numerics/native_srix_backend`
