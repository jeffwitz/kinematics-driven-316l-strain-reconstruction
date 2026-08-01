# P43 (ell, alpha) matrix and selection — results

Date: 2026-08-01
Preregistration: `p0043_small_parameter_matrix_preregistration.md`, validated
2026-08-01 with corrections C1 to C4 and amendments A1 to A4.
Indicator validation: `p0043_indicator_validation_results.md`, passed before the
matrix was read.
Machine-readable results:
`reference_data/p0043_small_parameter_matrix_v1/<profile>/selection_report.json`.

## Registered outcome: case B, a zone

**No parameterisation is selected.** Ten of the fourteen usable points form an
indistinguishable zone, the most frequent minimax winner takes only `32.9 %` of
the bootstrap draws, and the registered bar for "preferred" is `80 %`.

The scientifically useful part is not the verdict but why it came out that way,
and what the individual indicators say underneath it.

## The matrix

| `ell` (um) | `alpha = 0.5` | `alpha = 1` | `alpha = 2` | `alpha = 4` |
|---:|:---:|:---:|:---:|:---:|
| 20 | ok | ok | ok | **no convergence** |
| 40 | ok | ok | ok | **no convergence** |
| 58.88 | ok | archived | archived | archived |
| 90 | ok | ok | ok | ok |

Fourteen of sixteen points usable. **Zero cutbacks on every point that
converged**: where the solve works it works cleanly, so the two failures are not
the end of a continuum of increasing difficulty but a boundary.

Both failures sit at maximum coupling and short range. That corner is where
`Achi = Hchi * ell^2` is smallest while `Hchi` is largest: a stiff penalty tying
`chi` to `p` with an almost local Helmholtz operator to spread it. Reported as a
solver-convergence property at these registered settings, nothing more.

Per amendment A3 the two points are excluded and their settings were **not**
retuned; retuning one point would make it incomparable with the fifteen others.
The iso-`Achi` pair at `1600` loses a member as a result.

## Primary profile, `legacy_script_2021`

| point | `D_shape` | `D_amplitude` | `D_localisation` | `D_presence` | `R` | `J_inf` |
|---|---:|---:|---:|---:|---:|---:|
| a0.5-ell20 | `0.686` | `0.217` | `0.287` | `0.035` | `0.97` | `0.918` |
| a0.5-ell40 | `0.672` | `0.137` | `0.271` | `0.184` | `0.83` | `0.879` |
| a0.5-ell58.88 | `0.666` | `0.100` | `0.266` | `0.251` | `0.78` | `0.872` |
| a0.5-ell90 | `0.668` | `0.056` | `0.264` | `0.315` | `0.73` | `0.875` |
| a1-ell20 | `0.668` | `0.129` | `0.292` | `0.229` | `0.80` | `0.875` |
| a1-ell40 | `0.656` | **`0.011`** | `0.294` | `0.452` | `0.64` | `0.859` |
| a1-ell58.88 | `0.655` | `0.049` | `0.285` | `0.561` | `0.57` | `0.857` |
| **a1-ell90** | `0.653` | `0.100` | `0.278` | `0.655` | `0.52` | **`0.855`** |
| a2-ell20 | `0.658` | `0.036` | `0.313` | `0.463` | `0.63` | `0.861` |
| a2-ell40 | `0.644` | `0.181` | `0.310` | `0.815` | `0.44` | `1.054` |
| a2-ell58.88 | `0.647` | `0.286` | `0.383` | `0.992` | `0.37` | `1.284` |
| a2-ell90 | `0.641` | `0.365` | `0.460` | `1.126` | `0.32` | `1.547` |
| a4-ell58.88 | `0.625` | `0.605` | `0.721` | `1.518` | `0.22` | `2.572` |
| a4-ell90 | `0.634` | `0.718` | `0.841` | `1.698` | `0.18` | `3.054` |
| homogeneous control | `0.763` | `1.919` | `1.000` | `3.977` | `0.02` | — |
| translated control | `0.794` | `0.237` | `0.415` | `0.773` | `0.46` | — |

`R` is the high-pass strain energy the candidate carries relative to the DIC.
`D_null` per indicator: `0.763` from the homogeneous control on shape, and
`0.237`, `0.415`, `0.773` from the translated control on the other three.

## The decisive finding: the rule is dominated by its weakest indicator

Dynamic range across the fourteen candidates:

| indicator | min | max | ratio | control |
|---|---:|---:|---:|---:|
| `D_shape` | `0.625` | `0.686` | **`1.10`** | `0.763` |
| `D_localisation` | `0.264` | `0.841` | `3.18` | `0.415` |
| `D_presence` | `0.035` | `1.698` | **`48.7`** | `0.773` |
| `D_amplitude` | `0.011` | `0.718` | **`65.7`** | `0.237` |

`D_shape` separates the best candidate from the worst by `10 %`, and separates
the best candidate from the **homogeneous control** by only `22 %`. Normalised,
`Z_shape` therefore sits between `0.82` and `0.90` for every single candidate.

Since the minimax keeps the **worst** normalised defect, shape sets `J_inf` for
**eight of the fourteen points**, and compresses the eight best into a band of
`2.8 %` while their presence defects differ by a factor of twenty.

**Case B is therefore not a statement that the candidates resemble one
another.** They do not: amplitude and presence separate them by factors of 66
and 49. It is a statement about the aggregation rule, which is governed by the
one indicator that does not discriminate.

## What the indicators say underneath the rule

**Coupling removes fluctuation energy, monotonically in both parameters.** The
presence map is smooth and ordered: `R` falls from `0.97` at
`(alpha = 0.5, ell = 20)` to `0.18` at `(alpha = 4, ell = 90)`. Every increase
in either `alpha` or `ell` costs high-pass strain energy that the DIC has and
the model then lacks.

**Amplitude has an interior optimum, presence does not.** The q95 ratio is best
at `(alpha = 1, ell = 40)`, `0.011`, with `alpha = 0.5` under-coupled and
`alpha >= 2` over-coupled. Presence, on the other hand, is monotonically best at
the least coupling and shortest range, that is closest to local. **The two
disagree about the direction to move**, which is the amplitude-versus-
fluctuation tension in its sharpest form.

**`alpha` is constrained; `ell` is not.** Minimax by row:

| `alpha` | `ell=20` | `ell=40` | `ell=58.88` | `ell=90` | span |
|---:|---:|---:|---:|---:|---:|
| 0.5 | `0.918` | `0.879` | `0.872` | `0.875` | `0.046` |
| 1 | `0.875` | `0.859` | `0.857` | `0.855` | **`0.020`** |
| 2 | `0.861` | `1.054` | `1.284` | `1.547` | **`0.686`** |

At `alpha = 1`, sweeping `ell` over a factor `4.5` moves the score by `2.3 %`.
At `alpha = 2` the same sweep moves it by `80 %`. **There is a real interaction:
the spatial range only matters once the feedback is strong.** At weak coupling
this observable does not see `ell` at all.

**The `Achi` degeneracy is refuted, and the rule hides it.** The iso-`Achi` pair
at `800`, `(alpha = 2, ell = 20)` and `(alpha = 0.5, ell = 40)`, separates
clearly on the indicators — amplitude `0.036` against `0.137`, presence `0.463`
against `0.184`, energy `0.63` against `0.83`. `Hchi` and `ell` therefore act
separately on this observable rather than only through their product. Their
`J_inf` values are `0.861` and `0.879`, nearly identical, because for both the
minimax is reading shape.

## Morphology, as a separate diagnostic

Computed outside the registered selection, since section 16.10 forbids opening a
new criterion line inside this campaign. It enters neither the front nor the
minimax.

With the DIC Otsu threshold frozen at `4.535e-03`:

| quantity | range over the 14 candidates | DIC |
|---|---|---|
| active fraction | `22.8 %` to `25.0 %` | `26.2 %` |
| object count | **1** for 13 of 14 | **2** |
| minor-axis ratio | `1.88` to `2.08` | `1` |

**No `(ell, alpha)` combination reproduces the two-band morphology.** All give a
single merged object with a minor axis about twice too large. Coupling barely
touches it: the ratio moves from `2.08` at `alpha = 0.5` to `1.88` at
`alpha = 4`, a `9 %` change for a factor `8` in coupling.

**The one apparent exception is an artefact, and it is the v2 defect
reproducing.** `a2-ell20` alone gives two objects and looks best on every
morphological descriptor. Its second object is `620 px` against the DIC's
`8 340 px` band. Objects are paired by rank, so the reported minor-axis ratio of
`1.05` is the mean of `1.90` for the real object and `0.20` for the speck. On
its real object it is `1.90`, like everything else.

In criteria set v2 the blind profile caught the same failure with a `413 px`
speck on the translated control. It reproducing here on a different field shows
it is a property of the descriptor, not of that control. **Any future use of
morphology must pair objects by spatial overlap or comparable size, never by
rank.**

Morphology does reject the controls: the homogeneous gives zero objects, the
translated a minor-axis ratio of `2.58` beyond any candidate. But between
candidates it is even less discriminating than `D_shape`, and would only add a
second low-dynamic indicator to dominate the minimax for the same reason.

## Answers to the six registered questions

1. **Does `alpha` act mainly on amplitude?** It acts on amplitude and on
   presence, both strongly and monotonically. Presence is the cleaner of the two.
2. **Does `ell` act mainly on spatial organisation?** No. At `alpha <= 1` it
   barely acts at all; at `alpha = 2` it acts on everything at once.
3. **Does the best `alpha` depend on `ell`?** Yes, weakly: the best row is
   `alpha = 1` at every `ell`, but the penalty for `alpha = 2` grows sharply with
   `ell`.
4. **Is there a valley rather than a point?** A valley, covering `alpha <= 1` at
   every tested `ell`.
5. **Is the preferred zone stable?** Stability across profiles is reported below.
6. **Can the local model or a control still beat the micromorphic solutions?**
   The controls do not: both are worse than every candidate on the indicators
   that discriminate. The local model was not part of this grid.

## Limits

P43 only, one loading path, one ROI. This is a provisional reconstruction
parameterisation question, and no parameterisation was selected anyway. It does
not demonstrate an internal length of 316L, does not demonstrate
transferability, and neither validates nor refutes the nonlocal formulation.
`D_self` is an upper bound on the measurement floor, so every `Z` is a lower
bound. The two non-converged points are a property of the solver at these
settings, not a claim about the physics.
