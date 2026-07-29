# P43 measured-history Newton line-search pre-registration

**Status:** fixed before implementation or execution.

## Motivation

The unchanged Newton solver rejects the state-3-to-state-4 measured boundary
transition after generating trial engineering strains of 82.257 and 58.011 in
two neighbouring upper-boundary elements. The measured incremental EVM is
only `3.057e-4`. This is a global Newton overshoot.

## Algorithm under test

Add an optional local-model backtracking line search after solving
`K du = -R`:

1. start with `lambda = 1`;
2. evaluate the trial residual from the last committed constitutive state;
3. accept when `||R(u + lambda du)|| <= (1 - c lambda) ||R(u)||`;
4. otherwise multiply `lambda` by `0.5`;
5. stop after 12 trials or below `lambda = 2^-12`;
6. if no trial is accepted, reject the load increment and use the existing
   cutback.

The Armijo coefficient is fixed to `c = 1e-4`. Constitutive failure counts as
a rejected line-search trial. No material state is committed during the line
search.

This first implementation is authorised only for local plasticity. Coupled
micromorphic use must raise an explicit error because a trial would otherwise
need to reconverge the non-local fixed point.

## Compatibility and acceptance

- the feature is disabled by default;
- with the feature disabled, all existing results and iteration counts are
  unchanged;
- on a converged proportional small case, enabling it must change final
  `U`, `S`, `E`, `PE`, `PEEQ`, and `RF` by less than `1e-10` relative;
- rejected trials must not contaminate committed MFront state;
- line-search evaluations, reductions, failures, and minimum accepted factor
  are reported;
- the measured P43 run is accepted numerically only if all 40 states converge
  without non-finite fields.

No constitutive parameter, material map, DIC state, tolerance, load step,
non-local parameter, or linear solver setting may be changed.
