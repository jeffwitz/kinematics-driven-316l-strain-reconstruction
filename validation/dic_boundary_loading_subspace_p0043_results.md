# P43 boundary temporal noise and loading subspace — stage 0 results

Date: 2026-07-30
Preregistration: `dic_boundary_temporal_regularisation_preregistration.md`.
Primary machine-readable result:
`reference_data/dic_boundary_loading_subspace_p0043_v1/report.json`.
No mechanics were run. The immutable history was not modified.

## Short answer

**The registered expectation failed.** State 4 is not an outlier of the
measured boundary history, and no measurement outlier explains the state 3 to
state 4 nonlinear failure. Per the preregistration, the boundary-noise
explanation is withdrawn and nonlinear globalisation returns as the leading
hypothesis for that failure.

Two results survive and are worth keeping:

- the measurement noise is `0.047--0.051 px` per state and is **almost entirely
  affine**, which independently confirms the archived repeated-frame bound by a
  completely different route;
- the boundary loading is genuinely one smooth mode carrying `99.91 %` of the
  displacement energy, so a low-dimensional temporal regularisation is the
  right architecture if stage 1 proceeds — but on a weaker justification than
  the one that motivated this campaign.

## The registered test and its outcome

Registered: *state 4 must keep `|z| >= 3` on at least one of the two scores when
scored against all 40 states.*

| Score at state 4 | Value | Registered bound |
|---|---:|---:|
| loading coefficient, second temporal difference | `0.13` | `>= 3` |
| affine transverse strain, second temporal difference | `1.66` | `>= 3` |

Both fail by a wide margin. The largest `|z|` reached by the loading
coefficient anywhere in the 40-state history is `1.99`, at state 3. The maximum
of 39 standard Gaussian draws is expected near `2.7`, so the loading path is in
fact **smoother than pure noise would allow**. There is no outlier to find.

The only state exceeding `|z| = 3` on any score is state 20, on the affine
strain (`z = -3.85`), with a loading-coefficient score of exactly `0.00`. It is
therefore not a loading anomaly, and in any case the solver never reaches it:
it fails at state 4.

Per-increment signal-to-noise on the affine transverse strain places state 4 at
`3.73`, above the median of `3.52` and ranked about eighteenth of forty. State
4 is one of the more ordinary increments of the history.

### Why the earlier six-state reading was wrong

`dic_multistep_p0043_boundary_outlier_results.md` archived six states, from
which state 4 looked like a `3.5 sigma` excursion. That estimate scored a raw
increment against the scatter of five neighbours of a *trending* series. The
increments accelerate over the history, so their raw scatter is dominated by
the trend, not by noise, and the apparent excursion was an artefact of the
short window. Second-differencing removes the trend, and a robust scale over 37
realisations replaces the five-neighbour estimate. The excursion disappears.

## Measurement noise

| Quantity | Value |
|---|---:|
| per-state noise, robust (MAD) | `0.0511 px` = `9.40e-5 mm` |
| per-state noise, RMS | `0.0470 px` = `8.65e-5 mm` |
| archived repeated-frame bound | `0.06283 px` |
| noise realisations used | 37 |

Both estimators are upper bounds: genuine temporal curvature inflates them.
They agree with the archived repeated-frame residual, which was obtained from a
repeated image pair rather than from the temporal structure of the history.
Two independent routes converging near `0.05 px` is the strongest statement the
project currently has about DIC noise on this ROI.

Validity check of the estimator: the lag-1 autocorrelation of the
second-difference series is `-0.561`, against `-2/3` for pure independent noise
and a positive value for a curvature-dominated series. The second differences
are therefore noise-dominated, which is what the estimator assumes.

### The noise is affine

| Quantity | Value |
|---|---:|
| median non-affine fraction of a noise realisation | `9.63 %` |
| maximum non-affine fraction | `38.2 %` |
| affine `eps_xx` noise, per state | `1.53e-05` |
| affine `eps_yy` noise, per state | `1.33e-05` |
| affine `gamma_xy` noise, per state | `3.76e-06` |
| affine `eps_xx` noise, per increment | `2.16e-05` |

About `90 %` of the measurement noise is absorbed by an affine fit. The noise
behaves as a coherent global jitter of the boundary — translation and uniform
stretch — not as node-level decorrelation.

This resolves the discrepancy flagged when this campaign was proposed: the
archived non-affine boundary residual (`0.0016--0.0033 px`) is `26x` below the
archived noise `sigma` not because the noise is small, but because the noise is
almost entirely in the affine band that the residual metric removes by
construction.

It also settles the Saint-Venant question. The band a padded interior partition
is protected against — short-wavelength, self-equilibrated content — is the
band that carries almost none of the noise. The band that carries the noise is
quasi-uniform and does not decay into the interior at all. Padding is not a
defence against this noise, and a spatial filter has essentially nothing to
remove.

## Loading subspace

| Mode | Energy fraction | Temporal roughness |
|---:|---:|---:|
| 1 | `0.999118` | `0.0023` |
| 2 | `0.000870` | `0.0539` |
| 3 | `1.22e-05` | `0.1087` |
| 4 | `1.03e-07` | `0.5811` |
| 5 | `7.83e-08` | `0.4714` |

One mode carries `99.91 %` of the boundary displacement energy and is
temporally smooth to `0.23 %` of a white series. Modes 1 to 3 carry
`99.999 %`. Beyond mode 13 the roughness sits at `1`, the white-noise value.

The registered falsifier — no mode below `R = 0.5`, or retained energy under
`90 %` — does not fire: retained energy is `99.99997 %`. The low-dimensional
loading model is supported. The boundary of P43 moves along a single smooth
monotone path, which is what a temporal regularisation of a handful of
coefficients would act on.

Known artefact of the selection rule: mode 41 is numerically degenerate
(singular value near `1e-18`) and its roughness is meaningless, so it is
counted as signal. It carries no energy and does not affect the conclusion, but
stage 1 must add a per-mode energy floor to the rule.

## Per-increment signal-to-noise

| Quantity | Value |
|---|---:|
| median | `3.52` |
| maximum | `6.81` |
| minimum | `0.62` (state 8) |
| increments below unity | 5 of 40 (states 5, 8, 19, 24, 25) |

The affine strain increments are mostly resolved. This corrects by a large
factor the order-of-magnitude estimate that motivated the campaign, which put
the increment signal-to-noise near `0.3` by assuming the noise mapped fully
into the affine gradient. Measured propagation through the same affine fit
gives `1.53e-05` per state, about seven times smaller than that estimate.

The accumulated-plasticity concern shrinks accordingly: a random walk of
`2.16e-05` over 40 increments reaches `1.4e-04`, against a final EVM RMS of
`3.78e-03`, so roughly `3.6 %` rather than the `18 %` estimated beforehand.

## Limitations

- states 31 and 32 were linearly interpolated by the archived repair, so their
  second differences are zero by construction and are excluded. The exclusion
  list is read from the immutable repair report, not from a magnitude cutoff;
- this campaign measures the **affine band on the boundary**. It does not
  measure element-level strain noise in the interior, which is a different
  quantity and remains unquantified;
- the noise estimators are upper bounds;
- image index is not a force-synchronised load fraction.

## Consequences

1. The state 3 to state 4 failure is **not** explained by the measured boundary
   history. As registered, nonlinear globalisation near the first local plastic
   transition returns as the leading hypothesis, and the Newton instrumentation
   deferred in favour of this campaign is reinstated: correction norm of the
   free degrees of freedom, element strain increment and tangent conditioning
   before the rejected constitutive trial.
2. Temporal regularisation is **not** withdrawn, but its justification changes.
   It is no longer a fix for the failure. It is a measured improvement on 5 of
   40 noise-dominated increments and a modest reduction of accumulated
   plasticity bias. Stage 1 must be re-registered on that basis, or deferred.
3. Any stage 1 keeps the low-dimensional architecture established here, with an
   added per-mode energy floor.

## Reproduction

```bash
fem-inhouse diagnose-dic-boundary-loading-subspace \
  --history validation/reference_data/dic_multistep_history_p0043_repaired_v1/repaired_history_mm.npy \
  --history-report validation/reference_data/dic_multistep_history_p0043_repaired_v1/report.json \
  --output validation/reference_data/dic_boundary_loading_subspace_p0043_v1 \
  --figure-output validation/figures/dic_boundary_loading_subspace_p0043_v1
```
