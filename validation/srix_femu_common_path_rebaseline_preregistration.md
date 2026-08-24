# E-SRIX-FEMU-COMMON-PATH-001R — preregistration

Date: 2026-08-24

## Objective

Rebuild the synchronized FEMU path after the correction of the first-step
Dirichlet initialization. The previous v9 path and its direct-vs-FD comparison
are historical diagnostics only; they are not a valid baseline for the
corrected forward problem.

## Fixed contract

- M8 twin (`pixels=8`), four logarithmic parameter perturbations, `h=3e-3`.
- The initial displacement is either absent or a predictor for free DOFs only;
  prescribed boundary values are always taken from the current load step.
- Search paths may use fail-fast Newton limits only to propose bisections.
- The final oracle uses the strict equilibrium tolerance and validates base plus
  all eight perturbations on one identical partition.
- A proposal path may come from the old v9 artifact, but is unqualified and is
  never reused as evidence.
- No P43, optimization, or constitutive-parameter identification is allowed.

## Acceptance gate

The new path is accepted only if all nine strict fixed-path replays converge.
Only then is the direct sensitivity compared with the common-path central FD:
each column must have relative L2 error `<2%` and cosine `>0.999`.

The old v9/PATH-002 artifacts are retained and marked superseded; no historical
file is overwritten.
