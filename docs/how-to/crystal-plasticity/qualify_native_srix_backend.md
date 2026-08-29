# Qualify the native SRIX backend

**Mode:** how-to  
**Domain:** crystal-plasticity

## Goal

Reproduce the registered native NumPy/Numba SRIX comparison against the archived
MFront P43 M20 reference. The native runner uses the same parameters,
orientations, boundary history and two local plane-stress closures; it does not
silently replace the MFront oracle.

## Prerequisites

Install the project with its numerical dependencies and make the archived M20
MFront fields available at
`validation/reference_data/p0043_m20_c_f_forward_identified_v1/fields_F.npz`.
The native runner reads the registered experimental inputs and the final
parameter point from the archived P43 report. No MFront shared library is
needed by this particular native-vs-archived-fields command, but the MFront
reference fields must exist.

## Run

Start with the nested native closure, then repeat with the coupled closure:

```bash
PYTHONPATH=.:src python scripts/run_p0043_m20_numpy_srix_forward.py \
  --pixels 20 --subdivisions 4 --local-iterations 30 \
  --local-linear-solver numba-lu12 \
  --plane-stress-solver nested \
  --coupled-block-solver numpy \
  --output validation/_generated/native_srix/p0043_m20_nested

PYTHONPATH=.:src python scripts/run_p0043_m20_numpy_srix_forward.py \
  --pixels 20 --subdivisions 4 --local-iterations 30 \
  --local-linear-solver numba-lu12 \
  --plane-stress-solver coupled \
  --coupled-block-solver numba-fused \
  --output validation/_generated/native_srix/p0043_m20_coupled
```

These are the two registered native closures. `nested` and `coupled` are
algorithms for the same three-traction local problem, not different physical
laws. The `numba-fused` choice affects the point-local implementation only.

## Expected outputs

Each output stem produces:

* `report.json` with crop, parameters, history, backend options and timing;
* `fields_numpy.npz` with displacement, stress, strain and scored observables;
* the MFront comparison metrics embedded in `mfront_comparison`.

The corresponding archived qualification reports are
`validation/reference_data/p0043_m20_numpy_srix_nested_v1/report.json` and
`p0043_m20_numpy_srix_coupled_v1/report.json`. The registered comparison
reports a displacement difference at approximately numerical round-off and a
plane-stress residual in the recorded tolerance; wall time is machine-
dependent.

## Verify

```bash
jq '{backend, crop, path_steps, scored_steps, parameters, timing,
     raw_rms_mm, mfront_comparison}' \
  validation/_generated/native_srix/p0043_m20_nested/report.json

jq '{backend, crop, path_steps, scored_steps, timing,
     raw_rms_mm, mfront_comparison}' \
  validation/_generated/native_srix/p0043_m20_coupled/report.json
```

For the local closure, inspect `plane_stress_residual_mpa` in the saved fields
or report diagnostics when present. Compare stress, elastic strain, signed and
accumulated slips, tangent/closure diagnostics and final displacement—not only
the displacement RMS or elapsed time.

## What this establishes

It reproduces the registered native nested/coupled implementation comparison on
the P43 M20 configuration and records the provenance needed to compare the
fields. The qualification ladder and its numerical tolerances are summarized in
{doc}`../../reference/evidence/native_srix_qualification`.

## What this does **not** establish

Performance is machine-, thread- and trajectory-dependent, so these commands
are not a universal speedup benchmark. Native/MFront field equivalence does not
identify SRIX parameters, qualify an experimental FEMU workflow or demonstrate
that a GPU backend already exists.

## See also

* {doc}`../../reference/numerics/native_srix_backend`
* {doc}`../../explanation/native-srix/optimization_strategy`
* {doc}`run_316l_crystal_plasticity`
