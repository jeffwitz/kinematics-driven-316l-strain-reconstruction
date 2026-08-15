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
