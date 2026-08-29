# Reproduce the registered full-Dirichlet spectral qualification

**Mode:** how-to  
**Domain:** spectral

## Goal

Reproduce the registered small full-Dirichlet spectral qualification and
inspect its mechanical diagnostics. This is the shortest reproducible route
through the matrix-free Newton--GMRES solver; it compares CPS4, two-history
TET2 and the one-history EBI-TET variant on the same registered case.

This page uses a qualification driver, not a generic arbitrary-DIC production
CLI. The generic formulation and Python API are documented separately; no
general-purpose spectral case runner is currently exposed here.

## Prerequisites

Install the project dependencies, build the registered MFront library and run
from the repository root:

```bash
TFEL_ENV_FILE=/path/to/tfel/env.sh \
./scripts/build_mfront_behaviour.sh
export MFRONT_BEHAVIOUR_LIBRARY="$PWD/build/mfront/src/libBehaviour.so"
```

The command constructs its registered 12x12 non-affine displacement history;
no prepared DIC directory is used by this qualification driver. The MFront
behaviour must be `fcc_forest_rubin_srix` (the default).

## Run

```bash
MFRONT_BEHAVIOUR_LIBRARY="$PWD/build/mfront/src/libBehaviour.so" \
PYTHONPATH=src python scripts/qualify_ebi_state_sharing.py \
  --mesh 12 --increments 8 --tolerance 1e-8 \
  --transform-backend scipy \
  --output validation/_generated/ebi_tet/state_sharing_m12_reproduced.json
```

The `--transform-backend` option selects the transform implementation used by
the spectral preconditioner. It does not change the nonlinear residual or the
constitutive update.

## Expected outputs

The command writes one JSON report containing:

* CPS4, TET2 and EBI relative errors (`errors`);
* transform and material-history provenance;
* Newton/GMRES counts and timings;
* verification residuals and side resultants.

No field NPZ or image is produced by this driver. The registered comparison
uses two constitutive histories for TET2 and one shared history for EBI.

## Verify

```bash
jq '{mesh, increments, behaviour, transform, material_states_per_pixel,
     errors, iterations, verification_residual}' \
  validation/_generated/ebi_tet/state_sharing_m12_reproduced.json
```

For the registered case, compare the report with the evidence thresholds: the
TET2 verification residual is at the recorded numerical tolerance, TET2 differs
from CPS4 by about `0.72%` in accumulated slip, and EBI differs from TET2 by
about `5.39%`. Preserve the exact JSON, source SHA and library SHA when
reproducing the result.

## What this establishes

It verifies execution of the registered full-Dirichlet spectral pipeline and
separates the supported two-history TET2 discretisation from the registered
EBI-TET state-sharing discrepancy.

## What this does **not** establish

Solver convergence is not constitutive validation, and the registered EBI
result is not a universal statement about every state-sharing formulation.
The DST-I/B0 operator is a preconditioner; it is not the physical nonlinear
residual solver. See {doc}`../../reference/numerics/newton_gmres_contract` and
{doc}`../../explanation/spectral_mechanics/index` for the contracts and
formulation.

## See also

* {doc}`../../how-to/reproduce/reproduce_tet2_qualification`
* {doc}`../../how-to/reproduce/reproduce_ebi_falsification`
* {doc}`../../reference/numerics/spectral_solver`
