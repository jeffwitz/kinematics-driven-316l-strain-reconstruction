# P43 measured-history Newton instrumentation — preregistration

Date: 2026-07-30
Prescribed by `Claude.md` section V6 and reinstated by
`dic_boundary_loading_subspace_p0043_results.md`, which refuted the measured
boundary-outlier explanation (`z = 0.13` against a registered `|z| >= 3`).

## Question

The measured-history solve fails on the transition from state 3 to state 4.
Why does one Newton correction produce a total engineering strain of `58` to
`82` in two adjacent upper-boundary elements when the measured strain there is
at most `8.89e-05`?

## What the archived failure already rules out

From `dic_multistep_mechanics_p0043_measured_repaired_v1/failure_report.json`:

| Event | Increment | Step size | Newton iteration | Max strain |
|---|---:|---:|---:|---:|
| first failure | 4 | `2.50e-02` | 7 | `82.26` |
| last failure | 14 | `2.44e-05` | 2 | `58.01` |

The step size falls by a factor `1024` across 11 cutbacks, and the failure
moves *earlier* in the Newton sequence, from iteration 7 to iteration 2.

A globalisation failure improves when the step shrinks: a smaller increment
gives a better initial guess and a more linear problem. This one gets worse.
A failure that is insensitive to step size, and worse at small steps, is not
a line-search or step-length problem. It points at the linear system itself.

The strain is also a **total** strain computed from the absolute displacement
field, not an increment. A value of `58` means the displacement field carries
about eighty element widths of relative motion across one element.

## Registered hypotheses

- **H1 near-singular or corrupted global tangent.** The correction
  `du = K^-1 (-R)` is enormous because `K` is ill-conditioned or has degenerate
  rows. Prediction: the tangent diagonal shows non-positive or near-zero
  entries, and `||du||` exceeds the prescribed boundary increment norm by many
  orders of magnitude, at the iteration before the constitutive rejection.
- **H2 corrupted constitutive tangent.** The plane-stress condensation returns
  a tangent far from the elastic operator for a few elements, which then
  poisons the assembled system. Prediction:
  `max|D_ep| / max|C_elastic|` is much greater than one. For J2 hardening the
  consistent tangent should never exceed the elastic operator in magnitude, so
  a ratio above `1 + 1e-6` is by itself a defect.
- **H3 genuine globalisation.** The corrections grow smoothly over iterations
  from a well-conditioned system. Prediction: `||du||` decreases or stays
  bounded as the step shrinks, and tangent diagnostics stay clean.

H3 is already weakened by the step-size behaviour above; it is kept so the
instrumentation can confirm or restore it.

## Instrumentation

Per Newton iteration, with no change to the numerical path when the trace is
disabled:

- increment, pseudo-time, step size, iteration index, outcome;
- norm of the prescribed boundary increment `||du_B||`;
- residual norm and relative residual;
- correction norm `||du||`, its maximum entry and the degree of freedom
  carrying it;
- ratio `||du|| / ||du_B||`;
- maximum absolute total strain entering the constitutive trial;
- tangent diagonal minimum, maximum and non-positive count;
- `max|D_ep|`, `max|C_elastic|` and their ratio.

Registered discriminator, evaluated at the iteration preceding the first
constitutive rejection:

| Observation | Conclusion |
|---|---|
| tangent diagonal has non-positive or near-zero entries | H1 |
| `max|D_ep| / max|C_elastic| > 1 + 1e-6` | H2 |
| both clean and `||du|| / ||du_B||` below `1e3` | H3 |

If H1 and H2 both fire, the constitutive tangent is reported as the upstream
cause and the global conditioning as its consequence.

## Constraints

- the trace is observational. No solver parameter, tolerance, predictor or
  step-control rule is changed in this campaign;
- the immutable history is not modified and no frame is removed;
- no result is accepted as converged. The run is expected to fail; the trace
  of the failure is the deliverable;
- the trace must not alter the numerical path. This is verified by rerunning a
  converging reduced case with and without the trace and requiring bitwise
  identical fields.

## Deliverable

`validation/dic_multistep_p0043_newton_instrumentation_results.md` and
`reference_data/dic_multistep_newton_trace_p0043_v1/`, containing the
per-iteration table and the discriminator outcome.
