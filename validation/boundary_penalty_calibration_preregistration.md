# Boundary penalty stiffness — calibration preregistration

Date: 2026-07-31
Written before any calibrated penalty solve has been run.

## Purpose

`boundary_enforcement="penalty"` returns `BOUNDARY_MISFIT`, the nodal gap
between the measured boundary displacement and the value the solver actually
imposes. That field is only interpretable if the spring stiffness `k` is fixed
by something other than taste, because the misfit is proportional to `1/k` and
any desired amount of disagreement can be manufactured by choosing `k`.

## The dimensional problem

The statistically natural weight for a measurement of standard deviation
`sigma` is `1/sigma**2`. That cannot be used directly: in

`Pi = Pi_int(u) + (k/2) * sum_B |u - u_measured|**2`

the term is an energy, so `k` is a stiffness in force per length, while
`1/sigma**2` has units of one over length squared. A normalisation is required
and it is not unique, so it must be registered rather than improvised.

## Registered rule: the discrepancy principle

`k` is chosen so that the solver disagrees with the measurement by exactly as
much as the measurement is uncertain:

> **RMS over boundary nodes of `|BOUNDARY_MISFIT|` = `sigma_measured`.**

`sigma_measured` is `0.0511 px`, that is `9.40e-5 mm`, the robust per-state
boundary noise measured in `dic_boundary_loading_subspace_p0043_results.md`.
It is an upper bound, so the calibrated `k` is a lower bound on stiffness and
the resulting misfit is an upper bound on disagreement.

This leaves **no free parameter** once `sigma` is known, and it is the same
principle already registered for the temporal smoother. It cannot be tuned on
convergence or on agreement with DIC, and doing so is forbidden here for the
same reason it is forbidden elsewhere in the project.

## Calibration procedure

The relation between `k` and the misfit is monotone and smooth: each mode of
the boundary relaxes by a factor `1 / (1 + k / K_eff)`. Calibration therefore
uses a secant iteration on `log k`, against the **elastic** operator only, so
one linear solve is needed per iteration rather than a full elastoplastic run.

The starting bracket is anchored on the median tangent diagonal of the boundary
degrees of freedom, which is the natural stiffness scale of the problem and is
already reported by the Newton trace.

The elastic calibration is declared an **approximation**. Plasticity softens
the interior and will move the achieved misfit. The full run therefore reports
its achieved RMS misfit so the deviation from target is visible and can be
stated, not hidden.

## Registered acceptance criteria

1. The calibration converges to within `5 %` of the target RMS misfit on the
   elastic operator.
2. The calibrated `k` is finite and positive, and the tangent diagonal shows no
   non-positive entry, so conditioning is not degraded.
3. On the reduced analytical case, a calibrated penalty solve reproduces the
   elimination solution to within the misfit target itself. A larger deviation
   would mean the penalty is perturbing the solution beyond the uncertainty it
   is meant to represent.

## Registered interpretation, and its limit

With `k` fixed this way, `BOUNDARY_MISFIT` becomes a map of **where the
mechanics and the measurement are incompatible at the noise level**. A node
whose misfit is at the RMS is unremarkable. A node whose misfit is several
times the RMS marks a location where the model cannot accommodate the measured
boundary without paying more than the measurement uncertainty.

What it does **not** establish is which of the two is wrong. A large misfit can
mean a bad measurement at that node, a model-form error nearby, or a genuine
mechanical feature the plane-stress J2 description cannot represent. The
indicator localises disagreement; it does not attribute it.

## Scope of this campaign

Implementation and calibration only. No production campaign is switched to
penalty enforcement, elimination remains the default, and no archived result is
recomputed. A scientific campaign using the indicator would be registered
separately.

## Deliverable

`validation/boundary_penalty_calibration_results.md` and a
`calibrate_boundary_penalty_stiffness` entry point.
