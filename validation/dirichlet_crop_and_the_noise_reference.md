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

## A pointwise whitener passes the null test — and moves the noise floor

`fem_inhouse.identification.pointwise_whitening`. Two stages: divide by the
standard deviation estimated at each point and component, then whiten the
remainder spectrally. Fitted on 300 **independent, non-overlapping** noise
patches and validated on 100 held out — the only check a self-consistent
whitener cannot fake.

| whitener | held-out norm | directional sd | worst direction |
|---|---:|---:|---:|
| spectral only (previous) | `0.998` on its own samples | — | fails at `8.78` |
| pointwise only | `0.919` | `0.963` | `1.405` |
| pointwise + spectral | **`0.990`** | **`0.979`** | `1.270` |

The composition returns noise for noise on data it has never seen, in norm and
along directions fixed before the samples were drawn. Neither stage alone does.

**And it shows the repeat-noise dataset understates the real noise floor.**
Applied to the experiment, state 1 comes back at `1.94` rather than `1.0`:

| state | 1 | 2 | 3 | 5 | 10 | 20 | 30 | 40 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| whitened residual / expected | `1.94` | `3.31` | `4.69` | `7.36` | `14.3` | `28.4` | `43.4` | `57.9` |

The repeat dataset images the same state twice; a history correlates *deformed*
states against a reference, where decorrelation is worse. A factor of two
between the two is unsurprising, and it is the history's floor that matters.

Taking state 1 as an empirical floor, the residual reaches about `30` times the
noise at state 40 — still a large, unambiguous signal.

Two caveats bound that number. State 1 may itself contain a little signal, which
would make the floor too high and the ratio conservative. And if decorrelation
grows with strain the floor grows too, which would make it optimistic. The
growth is close to proportional to the load and far too regular for noise, which
is what argues it is a real response; separating a load-proportional systematic
from plasticity is the question that remains, and it is now asked against a
metric that has passed its null test.

## Corrected: a plastic field of realistic amplitude explains the gap

The previous section is **superseded**. Its Krylov space was compressed by an
SVD of the responses `A Phi`, which reorders the directions by singular gain --
reintroducing exactly the observability ranking the residual-driven
construction was meant to avoid. The leading direction was therefore not
`A^T r`, which is why rank 1 gave a meaningless `0.9997`. And cutting at rank 32
answered a question nobody asked.

`scripts/qualify_plastic_amplitude_tradeoff_p43.py` removes the reordering and
runs LSQR on the same operator, with the iteration count as the regularisation.
State 40, referenced to state 20:

| iterations | relative residual | plastic RMS | plastic peak | correlation with the residual |
|---:|---:|---:|---:|---:|
| 1 | `0.791` | `5.8e-5` | `6.0e-4` | `+0.061` |
| 8 | `0.360` | `2.9e-4` | `1.5e-3` | `+0.125` |
| 32 | `0.127` | `5.6e-4` | `2.0e-3` | `+0.194` |
| 128 | `0.035` | `8.0e-4` | `3.3e-3` | `+0.323` |
| 512 | `0.010` | `9.8e-4` | `4.2e-3` | `+0.399` |

**The gap closes, and at an amplitude the experiment comfortably exceeded.**
Ninety-nine per cent of the residual is removed by a plastic field of RMS
`9.8e-4` and peak `4.2e-3`, against an archived accumulated plastic RMS of
`5.67e-3` and a measured peak equivalent strain of `1.1e-2`. Nothing implausible
is being asked of the material.

**Where to stop.** The residual at state 40 sits about thirty times the
empirical noise floor, so it can legitimately be reduced by that factor and no
further: a relative residual of about `0.033`, reached near **128 iterations**.
There the plastic field has RMS `8.0e-4` and peak `3.3e-3`. Past that, LSQR is
fitting noise, which is what the growing amplitude beyond 128 shows.

**And it does sit where the discrepancy is.** Compared with the residual it is
meant to explain -- not with the total DIC strain, which measures
`e_el + e_p` and not `p` -- the correlation rises monotonically from `+0.06` to
`+0.40`, and the share inside the residual's top decile from `0.124` to `0.148`
against `0.10` for an unstructured field.

So the earlier negative was an artefact of the ranking, compounded by an
arbitrary cut. A plastic eigenstrain field of realistic amplitude does explain
the post-elastic DIC discrepancy on this crop.

**Not yet done, and it is the question actually asked:** the baseline here is
the elastic extension, not Ludwik. Answering "can a freed plastic description
close the gap Ludwik leaves" needs the Ludwik field on M100 and the correction
operator around it, and that replay does not exist yet.

### The required plastic field grows monotonically with the load

Same LSQR, all four states, referenced to state 20:

| state | 32 iters | 64 | 128 | `p` RMS at 128 | `p` peak at 128 |
|---:|---:|---:|---:|---:|---:|
| 25 | `0.1195` | `0.0603` | `0.0302` | `1.84e-4` | `7.1e-4` |
| 30 | `0.1666` | `0.1056` | `0.0633` | `4.96e-4` | `3.26e-3` |
| 35 | `0.1377` | `0.0796` | `0.0446` | `6.57e-4` | `3.96e-3` |
| 40 | `0.1273` | `0.0671` | `0.0352` | `7.97e-4` | `3.30e-3` |

The states are solved **independently** — nothing imposes irreversibility, history
consistency, or even continuity of the flow direction — so the ordering is a
result, not a constraint. The RMS is strictly increasing, `1.84e-4` to `7.97e-4`,
and the gap closes to between `3 %` and `6 %` at every state, so state 40 was
not an accident. The correlation with the residual grows with the iteration
count at every state, reaching `+0.31` to `+0.35`, and the top-decile share
`0.14` to `0.15` against `0.10`.

The peak is monotone to state 35 and dips slightly at 40; the RMS, which is the
robust measure, is not affected.

### Two things this does not say

`p_RMS` here is the norm of a **tensor eigenstrain between two states**, while
the archived `5.67e-3` is an **accumulated scalar history variable**. For a
roughly proportional monotonic path the orders of magnitude are comparable, and
that is the whole claim; they are not the same quantity.

And 128 iterations is a *physically reasonable* stopping point matching the
estimated noise level, not a proven onset of noise fitting — the factor of
thirty behind it still carries the reservations recorded above. The conclusion
does not depend on it: at 64 iterations, well before that threshold, `93 %` of
the gap is already gone at `p_RMS = 6.9e-4`.

### The Ludwik baseline is blocked on a discretisation mismatch

The question actually asked is whether a freed plastic description closes the
gap *Ludwik* leaves, not the gap the elastic extension leaves. That needs the
Ludwik field on M100, and `solve_ebi_dirichlet_plane_stress` can produce it —
but it uses `EBITwoTriangleKinematics2D` with one material state per pixel,
while every operator built here uses `TwoSubcellDiagnostic2D` with two. The
strain fields are not comparable term by term.

Reconciling them is the next piece of work and it is not a detail: either the
residual machinery moves to the EBI kinematics, or the Ludwik solve is repeated
on the two-sub-cell one. Doing it by interpolation between the two would put an
uncontrolled error exactly where the measurement is being made.
