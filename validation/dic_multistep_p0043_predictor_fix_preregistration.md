# P43 elastic predictor fix — preregistration

Date: 2026-07-30
Cause established in `dic_multistep_p0043_newton_instrumentation_results.md`.

## Defect

`FixedCSRAssembler.assemble` returns the same CSR object with `matrix.data`
overwritten in place. In `run_fem`, `KII_el` and every `K_tang` come from the
same assembler, so `K_tang is KII_el`. The measured-history elastic predictor
`solve_el(-KIB_el @ du_B)`, called inside the increment loop, therefore solves
against whatever elastoplastic tangent was assembled last.

## Fix

Keep an independent copy of the elastic operator for the predictor, so that
`solve_el` is unaffected by later assemblies. The change is confined to the
ownership of that matrix. No tolerance, step rule, predictor formula, line
search or constitutive setting is modified.

## Registered acceptance criteria

Ordered; each must hold before the next is read.

1. **No archived result moves.** The full suite passes with the real MFront
   environment, and the reduced analytical case reproduces its archived
   stress and PEEQ values. The proportional path never called `solve_el`
   inside the loop, so any change there would itself be a defect.
2. **The predictor becomes an elastic predictor.** On the measured history, the
   trial strain at iteration 1 of increment 4 must fall from `1.855e-02` to the
   order of `1e-03`, consistent with a linear extrapolation from increment 3.
3. **The frozen-operator signature disappears.** No sequence of increments may
   fail at iteration 1 with a trial strain proportional to `dt`.
4. **Progress past the blockage.** The measured history must pass the state 3
   to state 4 transition that has blocked every previous attempt.

## Registered outcomes

- If criteria 1 to 3 hold and 4 holds, the multistep blockage is resolved and
  the run continues to whatever the next genuine limit is. Reaching state 40 is
  **not** promised and is not required for the fix to be accepted.
- If criteria 1 to 3 hold but 4 fails, the defect is fixed and a second,
  independent cause remains. It is documented as such, not patched further in
  this campaign.
- If criterion 1 fails, the fix is withdrawn and re-examined.

No convergence-driven tuning is permitted. If the run still fails, the failure
is archived with its new trace; the smoothing and line-search routes are not
reopened in this campaign.

## Claim boundary

Convergence of the measured history would establish that the solver can follow
it. It would say nothing about whether the reconstructed fields are physically
meaningful, which stays governed by the observation-operator and DIC-noise
results already recorded.
