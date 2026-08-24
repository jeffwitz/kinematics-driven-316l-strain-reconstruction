# SRIX-REGM — cold-start worklog and gate record

Date opened: 2026-08-23  
Branch: `agent/plastic-observability`  
Status: **CLOSED/NO-GO: REGM surrogate branch stopped; direct FEMU sensitivity is next**

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
| 5 | exact REGM/FEMU ranking | passed: `rho=0.866`, log-`r=0.878`, top-5 `3/5` |
| 5b | observed-space REGM/FEMU ranking | failed at transfer-only level |
| 6 | P43 authorization | **NO-GO** under the frozen observed-space rule |

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

Gate 5 exact passes on all 20 forward candidates: Spearman `0.866`, logarithmic
Pearson `0.878`, top-five overlap `3/5`. Median REGM/FEMU cost is
`2.643/13.047 s`, or `4.94x`; complete-forward cost varies from `6.44` to
`146.75 s` with adaptive convergence.

The necessary observed-space repeat is negative. Transfer-only gives
Spearman `0.326`, log-Pearson `0.276`, and top-five overlap `2/5`; transfer,
noise and whitening formally passes, but its FEMU objective has coefficient of
variation only `9.1e-5` because noise dominates. The frozen rule required both
levels to pass. See `validation/srix_regm_femu_observed_ranking_results.md`.

**Stop here.** Do not launch P43-A/M100 or optimize SRIX on measured data with
this objective. The REGM surrogate branch is now closed: `K0`, `K_alg`,
causal one-correction replay, and the corrected cumulative endpoint observable
all fail to reproduce the FEMU sensitivity geometry. The next method must use
the exact forward FEMU residual and tangent action; it must not reuse
`TensorPlasticObservabilityOperator`, `weak_equilibrium_residual`, or
`_assemble_sparse_stiffness` as its global sensitivity operator.

## Observation-placement ablation (2026-08-24)

The suspected placement error was tested before any new P43 work. The
pre-registration is `validation/srix_regm_observation_placement_preregistration.md`;
the machine-readable result is
`validation/reference_data/srix_regm_observation_placement_v1/report.json`.

The 20 existing candidates were replayed with raw or affine-preserving-
transferred histories, and the pseudo-displacement was scored with identity,
periodic FFT, or affine-preserving transfer. Against the same observed FEMU
target, raw replay retained the ranking (`rho=0.950` periodic,
`rho=0.940` affine), while transferred replay failed before scoring
(`rho=0.338` identity). The existing combined path was `rho=0.326`; its
periodic-score counterpart was `rho=0.290`.

At the true parameters, the transferred-input path generated a pseudo-
displacement RMS of `4.067e-7 mm`. The candidate trajectory spread was
`3.495e-8 mm`, ratio `11.64`. This localizes the dominant failure to
replaying `O(u*)` through the nonlinear SRIX history, rather than applying `O`
to the correction. The NO-GO before P43 remains in force.

## Latent-mode upper bound (2026-08-24)

On the exact M8 twin, a snapshot POD of `u* - O(u*)` was progressively added
back before the SRIX replay, while the score remained `O(delta_u)`. No new
FEMU solve was launched. The observed ranking rose from `rho=0.326` at rank
zero to `0.577`, `0.708` and `0.859` at ranks 3, 4 and 5; full rank 162 gave
`0.940`. Rank 5 recovered `99.9897 %` of the missing-history energy,
log-Pearson `0.888` and top-five overlap `4/5`.

This is a twin upper bound, not a real-DIC reconstruction: the POD basis uses
the exact latent history. The next gate must construct weak modes from the
observation, elastic reconditioner and noise model without using `u*`, then
repeat the ranking test. P43 remains blocked.

## Mechanical projection test (2026-08-24)

The preregistered first-order projection test is in
`validation/srix_regm_mechanical_projection_preregistration.md`; its primary
report is `validation/reference_data/srix_regm_mechanical_projection_v1/report.json`.
The transferred M8 history was corrected once or twice with the existing
`-K0^-1 B^T sigma` update, using damping 0.25, 0.50 and 1.00. The ranking stayed
at Spearman 0.326--0.341 and top-five overlap 2/5, although the truth replay
RMS decreased from `2.132e-7` to `1.381e-7 mm` at the most aggressive case.

Conclusion: a simple equilibrium projection reduces the residual but does not
restore the parameter-discriminating information. It is rejected as the next
P43 method. A constrained observation/mechanics projection remains a possible
twin-only experiment; P43 remains blocked.

## Information geometry diagnostic (2026-08-24)

The preregistered comparison is in
`validation/srix_regm_information_geometry_preregistration.md`; the primary
artifact is `validation/reference_data/srix_regm_information_geometry_v1/report.json`.
Three Jacobians were compared at the M8 truth: exact REGM, transferred REGM and
complete observed FEMU. The normalized spectra are respectively
`(1, .422, .0324, 4.65e-5)`, `(1, .337, .0178, 1.27e-5)` and
`(1, .542, .407, .0679)`. The FEMU condition number is `14.7`, against
`2.15e4` and `7.90e4` for REGM.

The leading two-dimensional exact-REGM/FEMU subspaces differ by `67.2 deg`,
while exact and transferred REGM differ by `0.81 deg`. Therefore the
observation operator is not the sole problem: REGM already loses local FEMU
sensitivity directions in exact kinematic space. P43 remains blocked; any
future REGM reformulation must pass this information-geometry gate before
identification.

### 2026-08-24 — Reconditionnement par tangente algorithmique

Le preregistrement est `validation/srix_regm_algorithmic_tangent_preregistration.md`;
l'artefact primaire est
`validation/reference_data/srix_regm_algorithmic_tangent_v1/report.json`.
Le résidu faible a été reconditionné à chaque état par la tangente
algorithmique consistante de SRIX, sans Newton global. Le spectre normalisé est
`(1, .37594, .03469, 8.62e-5)`, avec un conditionnement `1,16e4`, contre
`(1, .42199, .03240, 4.65e-5)` et `2,15e4` pour le `K0` élastique.

L'angle principal avec la géométrie FEMU observée passe de `68,4` à `73,9`
degrés; les angles de rang deux sont `67,75` et `11,50` degrés. Le remplacement
de `K0` par `K_alg` améliore donc modestement le conditionnement mais ne
restaure pas les directions paramétriques faibles. L'hypothèse « le défaut
vient seulement du préconditionneur élastique » est rejetée sur ce jumeau.
Le prochain diagnostic est le rejeu séquentiel à une correction par incrément,
avec réévaluation et commit causal; P43 reste bloqué.

### 2026-08-24 — Rejeu séquentiel à une correction par incrément

Le preregistrement est `validation/srix_regm_sequential_one_newton_preregistration.md`;
l'artefact primaire est
`validation/reference_data/srix_regm_sequential_one_newton_v2/report.json`.
À chaque incrément, le prédicteur est avancé depuis l'état corrigé précédent,
une correction avec la tangente algorithmique est calculée, puis le matériau
est réévalué et committé sur le déplacement corrigé. Il n'y a pas de Newton
global convergé.

Le spectre normalisé devient `(1, .56251, .05764, 2.30e-4)` et le
conditionnement `4,35e3`. C'est une amélioration par rapport au REGM à histoire
fixe, mais l'angle de rang deux avec FEMU reste `67,91` degrés et les deux
directions faibles FEMU (`.4067, .0679`) ne sont pas retrouvées. Le gate de
géométrie est donc négatif. La séquence des diagnostics REGM peu coûteux est
close : ne pas lancer P43; toute méthode suivante doit repasser un jumeau et un
classement REGM/FEMU indépendants.

### 2026-08-24 — Correction de l'observable séquentielle

Le premier rejeu séquentiel ne score que la correction du dernier incrément.
Cette quantité n'est pas l'écart de déplacement à un endpoint FEMU. Le test
complémentaire `validation/reference_data/srix_regm_sequential_one_newton_v3/report.json`
conserve donc cette observable et ajoute
`accepted - displacement_history` avant le même transfert affine-preserving.

Le spectre de la correction seule reste `(1, .56251, .05764, 2.30e-4)`;
celui de l'écart cumulé devient `(1, .46460, .09381, 2.17e-4)`. Le troisième
mode est partiellement relevé, mais les directions FEMU `(0.4067, .0679)` ne
sont pas retrouvées et l'angle principal cumulé est `74.67 degrés`. La
correction méthodologique est donc réelle, mais elle ne change pas le
NO-GO du rejeu séquentiel. L'artefact détaillé est
`validation/srix_regm_sequential_one_newton_cumulative_results.md`.

## Final REGM decision and next gate (2026-08-24)

`E-SRIX-REGM-009` closes the surrogate branch. The cumulative endpoint score
was the correct observable for comparison with FEMU, but its spectrum was
`(1, .4646, .0938, 2.17e-4)` versus FEMU `(1, .5415, .4067, .0679)`.

The reason the theoretical one-Newton argument cannot be applied to that test
is now explicit: the sequential script uses the REGM mechanical discretization
(`TensorPlasticObservabilityOperator`, `weak_equilibrium_residual`, and
`_assemble_sparse_stiffness`), not the matrix-free residual/tangent action of
the M8 forward solver. Exact constitutive tangents and causal state updates do
not repair a different global operator.

The next and only authorized gate is `E-SRIX-FEMU-DIRECT-001`: one converged
M8 FEMU trajectory, persistent constitutive shadow histories, and four
right-hand sides solved with the exact `TraditionalTwoStateTriangleBatch` /
`solve_two_state_dirichlet_plane_stress` tangent action and boundary packing.
No P43, optimization, new REGM variant, or analytical MFront derivative is
authorized before the shadow method reproduces the archived FEMU FD columns.
