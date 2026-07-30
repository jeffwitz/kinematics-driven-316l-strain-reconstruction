# P43 elastic predictor fix — results

Date: 2026-07-30
Preregistration: `dic_multistep_p0043_predictor_fix_preregistration.md`.
Run: `reference_data/dic_multistep_predictor_fix_p0043_v1/`, git `6c4d44c`,
`4083.7 s` (`68.1 min`), 469 Newton iterations recorded.

## Short answer

**The measured 40-state boundary history now runs to completion.** Status
`completed_local_measured_boundary_history`. All four registered criteria pass.

The blockage that resisted frame removal, a state bridge, a `2 h 47` line
search and a boundary-noise hypothesis was one aliased CSR buffer.

## Registered criteria

| # | Criterion | Result | Verdict |
|---|---|---|---|
| 1 | no archived result moves | reduced analytical case identical before/after; 423 tests pass | **pass** |
| 2 | increment 4 iteration 1 trial strain falls to order `1e-03` | `1.855e-02` → `5.440e-04` | **pass** |
| 3 | frozen-operator signature disappears | 3 rejections, at iterations 10, 4 and 6; none at iteration 1; none scaling with `dt` | **pass** |
| 4 | pass the state 3 to state 4 transition | passed, and the full history completed | **pass** |

Criterion 4 was registered as "pass the transition"; reaching state 40 was
explicitly not promised. It was reached anyway.

## Increment 4, the transition that blocked every previous attempt

| Iteration | Before fix | After fix |
|---:|---:|---|
| 1 | `1.855e-02` | `5.440e-04`, corrected |
| 2 | `6.959e-02` | corrected |
| 3 | `3.069e-01` | corrected |
| 4 | `1.345` | **converged** |
| 5 | `5.598` | — |
| 6 | `2.200e+01` | — |
| 7 | rejected at `8.226e+01` | — |

The predicted value from a linear extrapolation of increment 3 was about
`8.7e-04`. The measured `5.44e-04` is of that order, which is what criterion 2
asked for. The predictor is an elastic predictor again.

## Whole-run diagnostics

| Quantity | Value |
|---|---:|
| converged increments | 65 |
| attempted increments | 68 |
| cutbacks | 3 |
| total Newton iterations | 469 |
| maximum Newton iterations in one increment | 10 |
| maximum `\|\|du\|\| / \|\|du_B\|\|` | `4.295e-01` |
| minimum tangent diagonal | `1.112e+02` |
| non-positive tangent diagonal entries | 0 |
| maximum `max\|D_ep\| / max\|C_elastic\|` | `1.000000` |

Corrections now stay smaller than the boundary increment that drives them. The
pre-fix run reached a ratio of `2.696` at increment 4 iteration 6 while
diverging geometrically.

The three surviving rejections, at increments 29, 50 and 59, occur at Newton
iterations 10, 4 and 6 with step sizes `2.500e-02`, `1.359e-02` and
`1.932e-02`. They are ordinary late-iteration overshoots recovered by a single
cutback each, not the pathological iteration-1 failures of the pre-fix run, and
their trial strains do not scale with the step size. Three cutbacks over 65
converged increments is normal elastoplastic solver behaviour.

## What this does and does not establish

Established: the solver can follow the measured 40-state boundary history of
P43 under the local J2/Ludwik model, and the earlier blockage had a software
cause with no physical content.

**Not established**: that the reconstructed interior fields are physically
meaningful. That question is unchanged and remains governed by the
observation-operator asymmetry and the DIC-noise results already recorded. In
particular:

- the states are ordered image indices, not a force-synchronised load fraction;
- 5 of 40 affine strain increments have signal-to-noise below unity, per
  `dic_boundary_loading_subspace_p0043_results.md`;
- no unloading branch exists, so no kinematic-hardening claim is licensed.

Deliberately **not** examined in this campaign: how the measured-path fields
differ from the archived proportional baseline. That comparison is the
scientifically interesting one and is exactly the kind of result that must be
preregistered before the numbers are read. It is left to a separate campaign.

## Consequences for the roadmap

- V6 item "impose the measured boundary displacement at every available step"
  moves from blocked to done;
- the conditional temporal prediction test, identifying on steps 1--20 and
  evaluating 21--40 with frozen maps, becomes executable;
- the temporal regularisation of stage 1, already weakened to a marginal
  improvement on 5 increments, is no longer needed to unblock anything. If it
  is pursued at all it is on its own merits.

## Reproduction

```bash
fem-inhouse --verbose run-dic-multistep-mechanics \
  --prepared-case data/processed/case_study \
  --source-campaign results/constitutive-local-p0043-pad150 \
  --history validation/reference_data/dic_multistep_history_p0043_repaired_v1 \
  --partition-id 43 --mode measured --record-newton-trace \
  --output validation/reference_data/dic_multistep_predictor_fix_p0043_v1
```
