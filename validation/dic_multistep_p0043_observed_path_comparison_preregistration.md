# P43 measured versus proportional path against DIC strain — preregistration

Date: 2026-07-30
Written before any comparison against DIC has been computed for either new run.

## Question

Is the final-state strain field reached under the **measured incremental
boundary history** closer to the DIC total strain than the one reached under a
proportional ramp to the same endpoint?

## The trap this campaign must avoid

`docs/explanation/current_evidence.md` and the V3 lot already established that
comparing a raw FEM field with an image-observed DIC field overstates the model
error, because only the experimental field passed through DISFlow. The V3
symmetric replay changed amplitude, morphology **and the ranking** of coupling
candidates, which is why micromorphic identification on the old objective was
suspended.

Repeating a raw comparison here and calling the answer would reproduce exactly
the error the project already documented and corrected.

**The registered primary comparison is therefore the symmetric one.** Each FEM
displacement field is warped onto the reference image and re-observed through
DISFlow with the same profile as the DIC, then EVM is reconstructed with the
same operator on both sides. The raw comparison is computed and reported as a
known-biased control, never as the conclusion.

## Fields

| Label | Path | Increments | Source |
|---|---|---:|---|
| A | measured 40-state DIC history | 40 | `reference_data/dic_multistep_predictor_fix_p0043_v1` |
| B | proportional ramp | 40 | `reference_data/dic_multistep_proportional40_p0043_v1` |
| C | proportional ramp, archived | 20 | `dic_symmetric_observation_p0043_v1`, case `local` |

C is already observed and archived; it provides continuity, not a control here.
The discretisation control for the A-B question was settled in
`dic_multistep_p0043_path_dependence_results.md`, where 40 against 20
increments moved core PEEQ by `0.20 %` against `15.82 %` for the path.

Profiles: `legacy_script_2021` as primary, `declared_medium_v4` as sensitivity,
matching the archived campaign so numbers stay comparable.

Metrics, on the core with padding excluded, identical to the archived symmetric
campaign: `relative_l2`, `rmse`, `pearson`, `top10_iou`, `absolute_q90_iou`,
`absolute_q90_active_fraction`.

## Registered significance margin

A difference between A and B counts as meaningful only if it exceeds the
95 % surrogate-sensitivity interval width already measured for that metric in
`dic_uncertainty_propagation_p0043_results.md` on the local case:

| Metric | Archived interval | Width, required margin |
|---|---|---:|
| relative L2 | `[0.4763, 0.4965]` | `0.0202` |
| Pearson | `[0.5933, 0.6118]` | `0.0185` |
| top-10 % IoU | `[0.2890, 0.3079]` | `0.0189` |
| absolute-q90 IoU | `[0.2978, 0.3195]` | `0.0217` |

Differences below these widths are inside the sensitivity of the metric to DIC
noise alone and are recorded as indistinguishable.

## Registered outcomes

- **Measured path closer**: A improves on B by more than the margin on the
  majority of the four metrics in the symmetric comparison, with no metric
  worsening by more than the margin.
- **Proportional path closer**: the mirror statement.
- **Split**: A improves beyond the margin on some metrics and worsens beyond it
  on others. This outcome is explicitly anticipated, since the archived work
  already found that the ranking of candidates depends on the objective. It
  must be reported as a split, not resolved by picking a favourite metric.
- **Indistinguishable**: no metric moves by more than its margin.

The verdict is read from the primary profile. The sensitivity profile is
reported and, if it disagrees, the disagreement is stated rather than averaged
away.

## Registered expectation

Recorded before computing: the measured path is expected to be **at least as
close** as the proportional one, because it is driven by the real boundary
history rather than a synthetic ramp. If it is instead **worse** beyond the
margin, that is a substantive negative result about the value of the measured
history and must be reported as such, not attributed to a bug without evidence.

## Claim boundary

DIC total strain is an image-derived observable, not ground truth. The states
are ordered image indices, not force-synchronised load fractions. Closeness to
DIC at the final state does not establish that the intermediate mechanical
states are correct, and no material parameter is identified or re-identified in
this campaign.

## Deliverable

`validation/dic_multistep_p0043_observed_path_comparison_results.md` and
`reference_data/dic_multistep_observed_path_p0043_v1/`.
