# P43 modal filtering of the measured boundary history — results

Date: 2026-07-30
Preregistration:
`dic_multistep_p0043_modal_boundary_filter_preregistration.md`.
Filtered history: `reference_data/dic_multistep_history_p0043_modal3_v1/`.
Mechanics: `reference_data/dic_multistep_modal3_p0043_v1/`.

## Short answer

Removing boundary content that sits `5.3` times **below** the measured noise
floor halves the Newton work and eliminates every cutback, perturbs core PEEQ
by `1.63 %`, and changes agreement with DIC by nothing measurable.

The path dependence survives the filter, so it is **not** a noise artefact.

## Filter acceptance

| Criterion | Value | Verdict |
|---|---:|---|
| removed content, RMS | `0.00972 px` | **pass**, `5.3x` below the `0.0511 px` noise |
| retained deviation energy | `99.989 %` | reported |
| origin and endpoint bit-identical | yes | pass |
| interior bit-identical | yes | pass |

Deviation-mode temporal roughness: `0.031`, `0.169`, `0.221` for modes 1 to 3,
then `0.561`, `0.512`, `0.660` for modes 4 to 6. The stage-0 roughness rule,
`R < 0.5`, **independently selects exactly three modes**. The rank fixed in
advance and the measured criterion agree without any tuning.

## Solver behaviour

| Run | Increments | Cutbacks | Newton iterations | Max per increment | Wall |
|---|---:|---:|---:|---:|---:|
| measured, unfiltered | 65 / 68 | 3 | 469 | 10 | `68.1 min` |
| **measured, 3-mode filter** | **40 / 40** | **0** | **245** | **8** | `34.8 min` |
| proportional ramp | 40 / 40 | 0 | 225 | 8 | `31.0 min` |

The registered expectation was that the filtered history should converge at
least as easily. It converges **essentially as smoothly as a synthetic
proportional ramp**: 245 Newton iterations against 225, no cutback, and half
the work of the unfiltered history.

The numerical difficulty of the measured history was therefore carried entirely
by content below the measurement noise floor. That is a strong practical
argument for the filter, and it is independent of any scientific claim.

## Core PEEQ

| Pair | relative L2 | Pearson | top-10 % IoU |
|---|---:|---:|---:|
| filtered vs unfiltered measured | `0.01630` | `0.99983` | `0.9844` |
| filtered vs proportional | `0.15359` | `0.98870` | `0.8714` |
| unfiltered vs proportional (archived) | `0.15822` | `0.98726` | `0.8631` |

The registered expectation was that the filter perturbs PEEQ by much less than
the `15.82 %` path effect. It perturbs it by `1.63 %`, about ten times less.

**The path dependence survives the filter**: `15.36 %` against the proportional
ramp, essentially the `15.82 %` measured before. The `15.8 %` reported in
`dic_multistep_p0043_path_dependence_results.md` is therefore not an artefact
of boundary noise; it is a property of the loading path itself.

Descriptive core PEEQ:

| Field | mean | p99 | max |
|---|---:|---:|---:|
| filtered | `3.2507e-03` | `2.5853e-02` | `7.2394e-02` |
| unfiltered | `3.2301e-03` | `2.5708e-02` | `7.3608e-02` |
| proportional | `3.0783e-03` | `2.3426e-02` | `6.4148e-02` |

**Explained on 2026-07-31**, in
`modal_filter_peeq_excess_results.md`: the filtered run accumulates `+0.64 %`
more mean PEEQ while its peak falls `1.65 %`. This is **redistribution, not
amplitude growth**. The filter removes scattered marginal yielding at low
levels — deciles 3 and 4 go negative and 217 fewer elements plastify — adds
plasticity across the band range, where the top four deciles carry `94.6 %` of
the excess, and shaves the extreme tail.

The sub-increment confound, 40 converged increments against 65, was originally
asserted here to be "of the right order to matter". That was written without
measurement and is **wrong**: calibrated on the zero-cutback proportional pair
it explains only `5.8 %` of the excess.

The band structure ratio of the filter's own effect is `3.95`, so the
perturbation is mildly band-concentrated, as expected when plasticity amplifies
a boundary perturbation. It is far below the `13.11` of the path effect.

## Agreement with DIC

Symmetric image-level observation, same margins as
`dic_multistep_p0043_observed_path_comparison_results.md`.

| Metric | filtered | unfiltered | proportional | filtered − unfiltered | margin | verdict |
|---|---:|---:|---:|---:|---:|---|
| relative L2 | `0.50406` | `0.50060` | `0.48515` | `+0.00346` | `0.0202` | indistinguishable |
| Pearson | `0.60231` | `0.60413` | `0.60390` | `−0.00183` | `0.0185` | indistinguishable |
| top-10 % IoU | `0.29219` | `0.29421` | `0.29866` | `−0.00202` | `0.0189` | indistinguishable |
| absolute-q90 IoU | `0.32062` | `0.32364` | `0.30876` | `−0.00302` | `0.0217` | indistinguishable |

The `declared_medium_v4` sensitivity profile agrees on every verdict; its
largest movement is `+0.00373` on relative L2.

The registered expectation of an indistinguishable result is met, and was
recorded in advance precisely so that this null would not be read as a failure.
This observable has now been shown insensitive to three separate changes of
this size: the loading path, and the boundary filter, on both profiles.

## Reading

The filter is justified by metrology alone: it removes content `5.3` times
below the measured noise, and the mode count it uses is the one the stage-0
roughness criterion selects independently. It was not chosen for its effect,
and its parameters were fixed before the mechanics ran.

What it buys is **numerical**, not evidential: half the Newton work and no
cutbacks, for a `1.63 %` perturbation of an unobservable internal variable and
no measurable change against DIC. That is a good trade for production, and it
should be described as such rather than as an improvement in fidelity.

## Claim boundary

Nothing here identifies or re-identifies a material parameter. Convergence
quality is not evidence of physical correctness. The filter removes content
below the noise floor of one measurement chain, characterised on one ROI; the
`0.0511 px` figure is itself an upper bound.

## Reproduction

```bash
fem-inhouse filter-dic-boundary-history \
  --history validation/reference_data/dic_multistep_history_p0043_repaired_v1/repaired_history_mm.npy \
  --history-report validation/reference_data/dic_multistep_history_p0043_repaired_v1/report.json \
  --rank 3 --output validation/reference_data/dic_multistep_history_p0043_modal3_v1

fem-inhouse --verbose run-dic-multistep-mechanics \
  --prepared-case data/processed/case_study \
  --source-campaign results/constitutive-local-p0043-pad150 \
  --history validation/reference_data/dic_multistep_history_p0043_modal3_v1 \
  --partition-id 43 --mode measured --record-newton-trace \
  --output validation/reference_data/dic_multistep_modal3_p0043_v1
```
