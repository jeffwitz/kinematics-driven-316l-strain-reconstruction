# SRIX-REGM — cold-start worklog and gate record

Date opened: 2026-08-23  
Branch: `agent/plastic-observability`  
Status: **Phase 0 audit complete; implementation gates in progress**

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

The Kelvin boundary is already centralized. `TwoSubcellDiagnostic2D` exchanges
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
| 1 | equilibrium-gap core and mechanical sign tests | pending |
| 2 | exact small SRIX twin, FD plateau and SVD | pending |
| 3 | deterministic theta4 recovery in identifiable subspace | pending |
| 4 | qualified transfer and noise degradation | pending |
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

## Next action

Implement Gate 1 in `src/fem_inhouse/identification/srix_equilibrium_gap.py`,
with unit tests independent of P43 and a thin twin qualification script. Do
not start an experimental identification campaign.
