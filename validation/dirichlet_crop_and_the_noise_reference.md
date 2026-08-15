# The residual is significant after all: the noise reference was wrong

Two objections were raised against the conclusion that the M100 field shows no
mechanical defect above the noise. Both were right that the analysis was
unsound. Both were wrong about which way the error pointed.

Scripts: `scripts/qualify_dirichlet_crop_observability.py`. Artefact:
`dirichlet_crop_observability.json`.

## The metric was hiding the signal

In a full Dirichlet problem the residual is pinned to zero on the boundary, so
its displacement norm is structurally small: a band of width `w` at plastic
strain `e` offsets the displacement by only `e w`. Differentiating recovers the
amplitude. Judged in displacement, the residual understates exactly the short
localised features being looked for; judged in strain it does not.

Measured, on the real history, the strain metric is about three times more
discriminating than the displacement one at every state.

## The noise reference was the wrong quantity

The residual is `r = (I - E P_b) u`, so under pure noise it is `(I - E P_b) n`,
not `n`. It was being compared against the raw DIC noise. Propagating real
noise realisations through the *same* crop-and-extend operator:

| | value |
|---|---:|
| propagated noise, displacement RMS | `5.18e-6 mm` |
| raw DIC uncertainty | `9.40e-5 mm` |
| ratio | **`18.2` times smaller** |
| propagated noise, strain RMS | `1.40e-4` |

The elastic extension absorbs most of the noise. The correct reference is
therefore some eighteen times smaller than the one used, so every residual was
that much **more** significant than reported, not less.

## What the experiment actually shows

Real P43 history, crop `(1580, 1030)`, `M_D = I`, 40 noise realisations:

| state | displacement / noise | strain / noise |
|---:|---:|---:|
| 1 | `0.16` | `0.30` |
| 5 | `0.48` | `1.02` |
| 10 | `0.64` | `1.74` |
| 20 | `1.21` | `3.41` |
| 25 | `1.70` | `4.38` |
| 30 | `6.92` | `8.15` |
| 40 | `4.91` | **`8.23`** |

The null test passes at state 1 — `0.16` and `0.30`, pure noise. The signal then
grows monotonically and reaches **eight times the propagated noise in strain**,
with a marked regime change between states 25 and 30.

**So the conclusion that there is no observable mechanical defect was wrong.**
There is one, it is large, and it appears where yield is expected.

## And the Dirichlet crop does not destroy it

A known eigenstrain band — 1 % amplitude, six pixels wide, shear-dominated —
was imposed on a 200-pixel domain under a far field, the crop taken from its
middle, and the procedure run on it with the noise propagated identically:

| guard | strain signal / noise | fraction of the injected strain recovered |
|---:|---:|---:|
| 0 | `20.5` | `0.573` |
| 10 | `17.7` | `0.577` |
| 25 | `17.7` | `0.613` |
| 50 | `16.2` | `0.649` |

The boundary does absorb part of the signature, as expected: `43 %` is lost at
the tightest crop. But `57 %` survives and would be detected at twenty times the
noise. Pushing the Dirichlet boundary away recovers only a little more —
`0.573` to `0.649` for fifty pixels of guard — so the guard band is a refinement,
not a rescue, and the tight crop was never the reason for the null result.

## What this corrects

The chain `M_D = I` → small residual → below the noise → no observable
plasticity had two broken links, and removing them reverses the conclusion. The
periodic-FFT artefact and the choice `M_D = I` both stand; what does not stand
is everything that was said about significance, in either direction, because it
was all measured against a reference eighteen times too large.

Every signal-to-noise figure in
`validation/dic_excitation_of_observable_plastic_modes.md` was computed against
that reference and is superseded here. The projections reported in sigma there
require the residual covariance `Q C_D Q^T`, not `C_D`, and have not yet been
recomputed.

## The strain-metric operator works; its whitener does not

`scripts/qualify_strain_metric_observability_p43.py`, artefact
`strain_metric_observability_m100.json`.

The observable was rebuilt as the **strain** of the residual, with the whitener
estimated from 200 real noise realisations pushed through the same
crop-and-extend operator. The operator itself is sound: its adjoint agrees to
`6.1e-14`, and the whitener reproduces the norm of the noise it was built from
to `0.9983`.

**But the null test fails.** At state 1, where the material is elastic and the
residual must be noise, the whitened residual reaches `1.80` times the expected
noise norm and a single mode reads `8.78` sigma. A whitener that cannot return
noise for noise is not measuring anything, so the spectrum and the z-scores from
this run are **not reported as results**.

The cause is the stationarity assumption. `DICSpectralWhitener` models one
spectral density for the whole field, and the pointwise standard deviation of
the propagated noise strain varies by a factor of `6.3` across the window. The
edge-versus-interior split is not where it varies -- that ratio is `1.03` -- so
the elastic extension is not what breaks it; the underlying DIC repeat noise is
simply not spatially uniform. Passing the self-consistency check while failing
the null test is exactly what a stationary model does when the truth is not:
it reproduces the global norm and misses the directional structure.

**What stands, therefore, is the unwhitened comparison of the previous section**,
which assumes nothing about stationarity: `0.30` at state 1 rising to `8.23` at
state 40, in strain, against noise propagated identically.

A usable whitener for this observable needs a pointwise variance, not a
spectral one, and ideally the local correlation as well. That is the next piece
of machinery, and until it exists no per-mode significance can be quoted in this
metric.
