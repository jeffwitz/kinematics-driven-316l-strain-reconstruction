# SRIX-REGM — cold-start worklog and gate record

Date opened: 2026-08-23  
Branch: `agent/plastic-observability`  
Status: **Gates 0--3 passed; Gate 4 negative; REGM/FEMU ranking running next**

This file is the single cold-start entry point for the reconditioned
equilibrium-gap identification of the FCC SRIX law. It is deliberately kept
separate from the historical TANN evidence. No TANN campaign is part of this
work.

## Scientific question

Can the information in measured kinematics, EBSD orientations and weak
mechanical equilibrium identify all or part of

```text
theta4 = (tau0, R, Q, b)
```

without a nonlinear global mechanical solve inside every objective
evaluation? A negative result, including a low-rank sensitivity matrix, is an
accepted scientific result.

## Frozen method

For each parameter vector, replay the measured strain history causally through
SRIX, commit between increments, assemble the interior weak residual, and form
the first-order displacement correction

```text
f(theta)       = B^T sigma(theta)
delta_u(theta) = -K0^-1 f(theta)
r(theta)       = W_D O(delta_u(theta)).
```

The sign is not accepted from notation alone: it must reduce the residual in a
mechanical unit test. DIC is already observed and is not filtered a second
time. No global Newton or Krylov solve is permitted in `residual_vector`.

## Phase 0 audit — verified contracts

| Need | Existing qualified implementation | Decision |
|---|---|---|
| nodal displacement to strain | `TwoSubcellDiagnostic2D.strain` | reuse exactly; output is engineering strain |
| weak stress residual | `TwoSubcellDiagnostic2D.divergence_from_sample_stress` | reuse; it returns `-w B^T sigma` |
| interior restriction | `pack_interior` / `unpack_interior` | reuse; boundary reactions do not enter the objective |
| elastic reference | `TensorPlasticObservabilityOperator.build` | reuse its once-only sparse assembly and factorisation |
| EBSD elastic reference | `rotated_plane_stress_stiffness` and `point_elasticity` | compare against isotropic `K0` on the twin |
| DIC transfer | `DICSpectralTransfer` | use the qualified legacy-profile/corrected-warp source |
| DIC noise | `DICSpectralWhitener` | reuse without changing its spectral floor |
| SRIX parameters | `resolve_srix_parameters` | immutable preset plus runtime overrides |
| plane stress | qualified SRIX batch factory | no closure in the identification driver |
| light response | `evaluate_in_plane_response(..., response_level="residual")` | no reconstructed tensors or returned tangent |
| transactions | `evaluate` / `commit` / `revert` | a fresh material batch per parameter evaluation |

The Phase-0 audit found and repaired one boundary bug before any twin result:
`rotated_plane_stress_stiffness` supplies engineering/Voigt stiffness, whereas
`TensorPlasticObservabilityOperator` stores Kelvin stiffness internally. The
public `point_elasticity` path now performs the same centralized conversion as
the isotropic path, and a dedicated test pins it.

The Kelvin boundary is now centralized. `TwoSubcellDiagnostic2D` exchanges
engineering strain and Voigt stress. `TensorPlasticObservabilityOperator`
converts only inside its Kelvin-specific plastic observability actions. The
new equilibrium-gap driver must therefore pass engineering strain directly to
the material and Voigt stress directly to the weak residual. No hand-written
factor of two or square root of two is allowed.

The existing `B0Green2D` is a nonlinear-solver preconditioner, not yet a
qualified replacement for the exact sparse `K0^-1` action required here. The
first scientific implementation therefore uses the factorised sparse operator.
A spectral backend is postponed until an equivalence and timing gate justifies
it.

## Gates

| Gate | Required evidence | Status |
|---|---|---|
| 0 | operator audit and cold-start record | passed |
| 1 | equilibrium-gap core and mechanical sign tests | passed |
| 2 | exact small SRIX twin, FD plateau and SVD | passed on M8 |
| 3 | deterministic theta4 recovery in identifiable subspace | passed on M8 |
| 4 | qualified transfer and noise degradation | negative: truth is not the objective minimum |
| 5 | REGM/FEMU ranking and timing comparison | pending |
| 6 | P43 authorization | blocked until Gate 5 GO |

## Required artefacts

- `validation/srix_regm_twin_preregistration.md`, written before the first twin
  result;
- `validation/reference_data/srix_regm_twin_v1/report.json`;
- `validation/srix_regm_twin_results.md`;
- `validation/reference_data/srix_regm_femu_ranking_v1/report.json`;
- figures generated only from the archived JSON/NPZ records.

## Stop rules

- Stop before P43 if the exact twin does not put the true parameters at the
  numerical floor.
- Stop before P43 if REGM does not rank parameter sets consistently with FEMU.
- Never choose a unique value along a numerically null singular direction.
- Never call a P43 result a 316L parameter identification without the final
  independent forward solve.

## Environment

Use `.venv/bin/python`. Before real SRIX tests, source
`/home/jeff/.local/share/tfel/env/env.sh` and set:

```text
MFRONT_BEHAVIOUR_LIBRARY=build/mfront/src/libBehaviour.so
SRIX_GENERIC_MFRONT_BEHAVIOUR_LIBRARY=build/srix-generic/src/libBehaviour.so
```

## Current quantitative result

Read `validation/srix_regm_twin_results.md` and its primary JSON. On the clean
M8 exact twin, the true/initial/identified REGM RMS values are respectively
`1.474e-13`, `3.143e-8` and `1.412e-13 mm`. The four parameters return to the
true valley with `0.248 %` projected log error. The four singular values span
`3.58e-6` to `1.66e-10`: the weakest direction is predominantly the
opposite-sign `Q/b` combination and is likely to be lost first under noise.

One REGM evaluation costs `2.90 s` versus `124.48 s` for the full twin
trajectory, a measured factor `43.0`. Constitutive replay consumes `94 %` of
the REGM time; `K0^-1` consumes about `1.5 %`. Optimizing the elastic inverse is
therefore not justified now.

The separate 32-state scaling benchmark in
`validation/srix_regm_scaling_results.md` measures `1.270 s` on M20 and
`19.708 s` on M100. M100 spends `18.693 s` in material replay and `0.447 s` in
`K0^-1`. The external 3D condensation still needs tangents internally for its
local plane-stress closure; compare the already-qualified native/GPS path
before attempting any new optimization.

## Next action

Gate 4 is archived in `validation/srix_regm_transfer_noise_results.md`. With
transfer alone, the true/initial/lowest-reached RMS values are respectively
`2.132e-7`, `2.310e-7`, and `1.300e-7 mm`; with measured noise and whitening
they are `1.741e-3`, `1.894e-3`, and `1.346e-3 mm`. The lower points are far
from the true SRIX parameters. The observation chain therefore preserves
sensitivity but biases the objective away from the generating parameters.

Run Gate 5 exactly as frozen in
`validation/srix_regm_femu_ranking_preregistration.md`. This gate asks whether
the exact-kinematics REGM objective still ranks laws like complete FEMU; it
does not retroactively repair the negative transferred/noisy recovery. Do not
start P43 unless the Spearman, log-Pearson and top-five gates all pass, and do
not use transferred/noisy REGM for unique parameter recovery without a revised
mechanically consistent observation treatment.
