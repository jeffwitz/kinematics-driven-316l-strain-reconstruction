# Boundary penalty stiffness — calibration results

Date: 2026-07-31
Preregistration: `boundary_penalty_calibration_preregistration.md`.

## Short answer

The spring stiffness is no longer a free parameter. It is fixed by the
discrepancy principle — the solver is allowed to disagree with the measurement
by exactly the measured noise, `sigma = 9.40e-5` mm — and the calibration
converges to that target in single-digit iterations.

The key verification: at the calibrated stiffness, the solution moves away from
the elimination solution by **the same amount as the misfit**. The penalty
therefore perturbs the mechanics by exactly the uncertainty it represents, and
no more.

## Registered criteria

| # | Criterion | Result | Verdict |
|---|---|---|---|
| 1 | calibration within `5 %` of target | `9.142e-5` mm against `9.40e-5`, `97.3 %` | **pass** |
| 2 | finite, positive, no non-positive diagonal | `k / K_ref = 2.7` | **pass** |
| 3 | penalty reproduces elimination to within the misfit | deviation equals misfit to `5 %` | **pass** |

## Calibration behaviour

On an analytic spring chain with a known stiffness scale, bisection on `log k`
reaches the target in **7 iterations**, giving `k = 5.477e5` against a reference
boundary diagonal of `2.0e5`, so `k / K_ref = 2.7`.

That ratio is the point of the exercise. A hard-Dirichlet impersonation would
need `k / K_ref` around `1e7`, which degrades conditioning; the earlier
uncalibrated experiment showed the displacement error rising again at
`k = 1e12`. The discrepancy principle instead lands on a spring of the same
order as the material it is attached to, which is well conditioned by
construction.

## Criterion 3: the penalty moves the solution by exactly the misfit

On the reduced analytical case, sweeping the spring:

| `k` | RMS misfit (mm) | `max abs(U_penalty - U_elimination)` (mm) |
|---:|---:|---:|
| `1e5` | `3.506e-6` | `3.674e-6` |
| `1e6` | `3.776e-7` | `3.980e-7` |
| `1e7` | `3.793e-8` | `3.998e-8` |

The two columns track each other to within `5 %` across three decades. The
deviation from the exactly constrained solution is the misfit, not some larger
uncontrolled perturbation. Calibrating the misfit to the noise floor therefore
bounds the perturbation of the mechanics at the noise floor as well.

Note that this reduced case sits far below the noise floor even at the softest
spring tested, because its displacements are small; the calibration would pick
a `k` near `4e3` here. The relation between deviation and misfit is what
generalises, not the absolute values.

## What the indicator now means

With `k` fixed this way, `BOUNDARY_MISFIT` is a map of where the mechanics and
the measurement are incompatible **at the noise level**. A node at the RMS is
unremarkable; a node several times above it marks a place where the model
cannot accommodate the measured boundary without paying more than the
measurement uncertainty.

It does **not** say which of the two is wrong. A large misfit can be a bad
measurement at that node, a model-form error nearby, or a real mechanical
feature that plane-stress J2 cannot represent. The indicator localises
disagreement; attributing it needs something else.

## Not done

The calibration has **not** been run on the P43 elastic operator, so the
production value of `k` for that ROI is not yet known. Doing so needs the
elastic stiffness assembled outside the Newton loop, which is plumbing this
campaign did not add. Until then the P43 indicator would have to be run at a
declared stiffness rather than a calibrated one, and its misfit read as
relative rather than absolute.

Elimination remains the default. No production campaign uses penalty
enforcement and no archived result was recomputed, as registered.

## Reproduction

```python
from fem_inhouse.core.penalty_calibration import calibrate_boundary_penalty_stiffness

result = calibrate_boundary_penalty_stiffness(
    elastic_stiffness_csr,
    boundary_dofs=mesh.dofs_bc,
    boundary_values=measured_boundary_displacement,
)
```
