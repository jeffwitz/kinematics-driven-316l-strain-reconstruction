# P43 matrix, section 9: indicator validation — results

Date: 2026-08-01
Preregistration: `p0043_small_parameter_matrix_preregistration.md`, validated
2026-08-01 with corrections C1 to C4 and amendment A1.
Machine-readable result:
`reference_data/p0043_small_parameter_matrix_v1/indicator_validation.json`.

**Run before the matrix was read, and committed before it finished computing.**
Which indicators survive therefore cannot be a reaction to the matrix. No
mechanics was rerun for this section; the (ell, alpha) campaign was still in
progress when these numbers were produced.

## Short answer

**All five acceptance criteria pass, and all four indicators enter the
selection.** Conclusions are stable across `32`, `49` and `96 px`: the verdicts
are identical at the three scales and the rank correlations against the
principal scale run from `0.88` to `0.98`.

One defect was found and repaired, in the measurement floor rather than in the
indicators. It is described below because it changed every normalised number.

## The defect the gate found: the floor was twelve times too large

Correction C2 registered `D_self` as the measured repetition residual, built
from the archived per-component deviations, `0.0403 px` in row and
`0.0624 px` in column, with the measured `38.2 px` coherence.

Synthesised that way, the residual produces an **EVM RMS of `1.64e-3`** — twelve
times the `1.363e-4` that the same measurement campaign reports for the same
repeated pair. The two archived quantities are not consistent with a Gaussian
field: the real residual is far smoother than a Gaussian field of that nominal
coherence, which matches what the chain report itself says the pair may contain,
slow optical drift.

Anchoring the floor on displacement amplitude therefore attributes to noise a
strain the measurement demonstrably does not produce. The indicators consume
strain, so **the residual is now calibrated on the measured spurious EVM RMS**.
The synthesised displacement deviation then comes out near `0.0034 px` instead
of `0.04`; that gap is the drift, not an error.

The effect was not marginal. With the displacement anchor the floor was
`D_presence = 0.737`, **worse than the local model's own `0.371`**, and two
acceptance criteria failed for that reason alone. Calibrated on strain, the
floor is `0.0039`.

| Floor `D_self` | shape | amplitude | localisation | presence |
|---|---:|---:|---:|---:|
| displacement-anchored, wrong | `0.424` | `0.345` | `0.500` | `0.737` |
| **strain-anchored, registered** | **`0.0045`** | **`0.0023`** | **`0.00092`** | **`0.0039`** |

Both the calibration and the twelvefold inflation are locked by tests, so the
reasoning cannot be silently lost.

`D_self` remains an **upper bound** on the floor: the archived pair is itself
described as a noise-and-drift upper bound. Every `Z` is therefore a lower
bound on the normalised defect.

## The nine registered cases at the principal scale

| Case | `D_shape` | `D_amplitude` | `D_localisation` | `D_presence` | `R` |
|---|---:|---:|---:|---:|---:|
| DIC against itself | `0` | `0` | `0` | `0` | `1.000` |
| repetition residual | `0.0045` | `0.0023` | `0.0009` | `0.0039` | — |
| amplitude `0.80` | `1.1e-16` | `0.223` | `0.168` | `0.446` | `0.640` |
| amplitude `1.20` | `0` | `0.182` | `0.125` | `0.365` | `1.440` |
| band displaced `16 px` | `0.800` | `0.027` | `0.062` | `0.040` | `0.961` |
| bands merged | `0.731` | `0.169` | `0.196` | `0.855` | `2.352` |
| band removed | `0.866` | `2.293` | `0.596` | `5.616` | `274.7` |
| band spurious | `0.962` | `3.019` | `0.691` | `4.847` | `127.4` |
| **homogeneous control** | `0.763` | **`1.919`** | **`1.000`** | **`3.977`** | **`0.019`** |
| **translated control** | **`0.794`** | `0.237` | **`0.415`** | `0.773` | `0.462` |
| local model, for scale | `0.724` | `0.320` | `0.332` | `0.371` | `1.449` |

## The five criteria, one by one

**Identity is optimal.** Exactly zero on all four indicators.

**The homogeneous control fails on presence and amplitude.** It carries
`1.9 %` of the DIC high-pass strain energy at the principal scale, giving
`D_presence = 3.98` against a floor of `0.0039` and against `0.371` for the
local model. Its `D_amplitude` is `1.92`. **This is the failure the earlier
gradient diagnostic could not catch**: there, a structureless field ranked
third of six because a fluctuation distance saturates near `1` for a candidate
with no content. The presence indicator exists for exactly this and does its
job. Its `D_localisation` is `1.000`, that is `FSS = 0`, so three of the four
indicators reject it outright.

**The translated control fails on shape and on localisation**, `0.794` and
`0.415`, both far above the floor. It is not caught by presence, `0.773`, nor
by amplitude, `0.237` — as expected of a control that has the right
microstructure in the wrong place.

**Amplitude and position are distinguishable.** A `1.20` rescaling gives
`D_shape = 0` exactly and `D_amplitude = 0.182`; a `16 px` displacement gives
`D_shape = 0.800` and `D_amplitude = 0.027`. Neither defect is mistaken for the
other, and the leak is a factor thirty below the signal.

**A removed band is worse than a moderate amplitude error**, `5.62` against
`0.365` on the worst indicator. **This one passes partly for an artefactual
reason and should not be over-read**: zeroing the fluctuation inside a
hard-edged corridor puts a step at the mask boundary, and the step multiplies
the high-pass energy by `275`. The registered rule already says only the sign of
the band-level cases is read. The criterion is satisfied, but by a margin that
the mask edge inflates.

## Stability across scales

Verdicts identical at `32`, `49` and `96 px`. Rank correlation of the case
ordering against the principal scale:

| Indicator | `32 px` | `96 px` |
|---|---:|---:|
| `D_shape` | `0.952` | `0.891` |
| `D_amplitude` | `0.915` | `0.879` |
| `D_localisation` | `0.952` | `0.976` |
| `D_presence` | `0.952` | `0.915` |

The registered wording asks for conclusions to stay qualitatively stable, not
for an identical ranking. Stability is therefore read as agreement of the
verdicts, with the rank correlations reported as the evidence behind that
reading; the first implementation demanded an identical ordering of ten cases,
which no near-tied set can satisfy and which would have failed on swaps that
change nothing anyone would conclude.

## Consequence for the campaign

No indicator is removed. The Pareto front and the minimax will be built on all
four, as registered.

Nothing here says anything about `ell`, `alpha`, or the micromorphic
solutions: no matrix point was read to produce it.
