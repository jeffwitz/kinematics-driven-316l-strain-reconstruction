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

## The maps: two metrics that disagree, and it matters

`scripts/plot_krylov_correction_maps_p43.py`, artefacts in
`krylov_maps_state40/` — `krylov_correction_maps.png`, `krylov_tradeoff.png`.

Drawn at state 40 referenced to state 20: the von Mises equivalent of the
measured strain increment, of the simulation as LSQR directions are added, their
difference, and the plastic field responsible. Native two-sub-cell layout,
averaged over sub-cells for display only, no interpolation. The forward operator
here is the **physical** one — the whitener belongs to the inverse problem, not
to the strain being looked at.

| iterations | whitened residual | **raw strain error** | equivalent-map correlation | plastic RMS |
|---:|---:|---:|---:|---:|
| 8 | `0.360` | `0.952` | `0.722` | `2.9e-4` |
| 32 | `0.127` | `0.881` | `0.767` | `5.6e-4` |
| 64 | `0.067` | `0.835` | `0.795` | `6.9e-4` |
| 128 | `0.035` | `0.784` | `0.825` | `8.0e-4` |
| 512 | `0.010` | `0.643` | `0.889` | `9.9e-4` |

**The two metrics disagree by a wide margin, and this qualifies the earlier
headline.** At 128 iterations the correction removes `96.5 %` of the residual in
the whitened metric but only `22 %` of the raw strain difference. At 512, `99 %`
against `36 %`.

The whitener divides by the pointwise deviation of the propagated noise and then
whitens spectrally. The elastic extension removes the smooth part of the noise,
so the propagated noise is comparatively small at low wavenumber and the
whitener weights those components heavily. The fit therefore concentrates on the
smooth part of the discrepancy and leaves the rough part largely untouched —
which is statistically correct for detecting signal against noise, and is not
the same statement as reproducing the measured strain field.

**What does improve steadily is the resemblance.** The correlation between the
simulated and measured equivalent-strain maps rises from `0.722` to `0.889`. The
maps show the correction building where the measurement localises rather than at
the edges.

So the honest reading, and the one to carry forward:

* a plastic field of realistic amplitude removes almost all of the
  **statistically significant** part of the discrepancy;
* it removes only a third of the **raw** strain difference, most of which sits
  in components the noise model treats as unreliable;
* and the corrected field looks progressively more like the measured one.

Whether the remaining raw difference is measurement noise the whitener is right
to discount, or real strain the noise model is wrong about, is the question the
Ludwik comparison and a better noise estimate have to settle. It should not be
asserted either way from these numbers.

## Without the whitener in the objective, the fit is exact — and cheap

`scripts/compare_raw_and_whitened_plastic_inversion_p43.py`, artefacts in
`raw_vs_whitened_state40/`: component maps for both objectives at 128 and 512
iterations, and the trade-off comparison.

Two inversions, same operator, same Krylov regularisation, same stopping points,
differing only in whether `W` appears in the objective. Both scored in both
metrics.

| objective | iterations | raw error | whitened error | `p` RMS | `p` peak |
|---|---:|---:|---:|---:|---:|
| raw | 8 | `1.1e-8` | `1.1e-8` | `1.374e-3` | `4.70e-3` |
| whitened | 128 | `0.784` | `0.035` | `7.97e-4` | `3.30e-3` |
| whitened | 1024 | `0.496` | `0.006` | `1.083e-3` | `4.13e-3` |

**The raw problem is solved exactly, in eight iterations.** Every component
follows: `e_xx`, `e_yy` and `g_xy` errors all fall to zero together, so there is
no component the correction cannot reach.

**The amplitude is the whole content of that result.** Exactness is guaranteed
in advance: `A` maps onto the strains of every zero-boundary interior field, and
the residual is one by construction, so *any* target — noise included — is
exactly reproducible. What is not guaranteed is the price, and the price is
`p_RMS = 1.374e-3` with a peak of `4.70e-3`, against an archived accumulated
plastic RMS of `5.67e-3` and a measured peak equivalent strain of `1.12e-2`.

So the question can be answered: **a mechanically admissible plastic field
reproduces the measured strain increment exactly, at an amplitude the material
comfortably exceeded.** That is the favourable scenario, not the one where
closing the gap demands an absurd field.

**And the whitened objective was indeed costing agreement.** At 1024 iterations
it uses a comparable amplitude, `1.083e-3` against `1.374e-3`, yet still leaves
half the raw strain difference. It is not spending its budget on the same thing.

The caveat is the one that matters and it is not small: an exact fit that
includes the noise is not an identification. What the raw inversion establishes
is a **bound** — the measured kinematics is mechanically admissible at a
plausible plastic amplitude — not that this particular field is the plastic
field. Separating the two needs the history constraint, which four independently
solved states do not provide, and the Ludwik comparison.

## A single subspace of rank 16 carries all four states

`scripts/qualify_shared_plastic_subspace_p43.py`, artefacts in
`shared_subspace/`.

Solving each state on its own is unconstrained enough that an exact fit is
guaranteed, so exactness there carries no information. Requiring all four states
to build their plasticity inside the **same spatial subspace** is the first
constraint that is neither a constitutive law nor a smoothness prior: the
material has one place where it yields, and the load changes how much. The
subspace is a block Krylov space driven by all four measured residuals at once,
kept in Krylov order and never reordered by singular gain.

| shared rank | raw error, states 25 / 30 / 35 / 40 |
|---:|---|
| 4 | `0.123` `0.092` `0.109` `0.115` |
| 8 | `0.0104` `0.0087` `0.0097` `0.0112` |
| 16 | `0.0001` `0.0001` `0.0001` `0.0001` |

**Sixteen shared spatial modes reproduce all four strain increments to `1e-4`**,
and eight already reach one per cent. That is 64 coefficients against 30 000
plastic components, with the four states forced through the same spatial
subspace — a genuine reduction, and the one the whole construction was aiming
at.

A consistency check passes: the shared subspace needs `1.37e-3` at state 40,
against `1.374e-3` for the independent exact fit of the previous section.

## But the coefficient history is not what plasticity should look like

| state | 25 | 30 | 35 | 40 |
|---|---:|---:|---:|---:|
| plastic RMS | `2.74e-4` | `1.56e-3` | `1.47e-3` | `1.37e-3` |

These are increments from state 20, so under monotonic loading they should grow.
They jump between 25 and 30 and then **decrease**.

The obvious explanation — that the exact fit is absorbing a state-independent
noise contribution which swamps the trend — does not survive the check: the same
non-monotone shape is already there at **rank 4**, where the fit is only 11 %
accurate and far from noise-limited. It is in the leading modes, not in the tail.

Two readings remain, and this run does not separate them. Either the plastic
increment relative to state 20 genuinely saturates after state 30, which would
be consistent with the regime change already located between states 20 and 25;
or the shared subspace is fitting something that is not plastic and happens to
be roughly constant over the later states.

Deciding needs the constraint this construction still lacks: nothing here
imposes irreversibility, and nothing ties `a_n` to `a_{n-1}`. A history-
constrained solve — coefficients monotone in the appropriate sense, or a
positive increment per step — is the next thing to try, and it is now cheap
because the subspace is only sixteen-dimensional.

## Irreversibility: the unconstrained fit is inadmissible, and the penalty test is malformed

`scripts/qualify_history_constrained_plastic_solve_p43.py`, artefacts in
`history_constrained/`.

Plasticity dissipates, so `sigma_k : (eps_p(k) - eps_p(k-1)) >= 0` at every
point between consecutive states. With the stress frozen at the unconstrained
solution the constraint is linear in the sixteen reduced coefficients.

**The first reading is a genuine result.** In the unconstrained exact fit,
**54.2 %** of the material points have *negative* dissipation. More than half
the domain is thermodynamically inadmissible, which is strong evidence that the
exactly-fitting field is not a plastic history whatever its amplitude.

**The second is a defect in my test, and it invalidates the rest of the table.**

| penalty | mean error | `p` RMS at state 40 | monotone | negative-power share |
|---:|---:|---:|---|---:|
| 0 | `0.0001` | `1.37e-3` | no | `0.542` |
| `0.01` | `0.839` | `3.98e-4` | yes | `0.618` |
| `1` | `0.994` | `7.9e-6` | yes | `0.617` |
| `1e4` | `1.000` | `8.0e-10` | yes | `0.616` |

A squared penalty on `min(power, 0)` has a **trivial minimiser at zero**: halving
the field quarters the violation without reorienting anything. The optimiser
takes that route — the amplitude collapses by four orders of magnitude, the
misfit goes to one, and the *fraction* of violating points does not improve at
all, drifting from `0.542` to `0.616`. The monotone amplitudes in the right-hand
column are an artefact of shrinking towards zero, not a history.

So this run does **not** show that irreversibility is incompatible with the
data. It shows that a penalty formulation cannot test it, because the constraint
is scale-free and the objective is not.

The fix is a hard feasibility constraint rather than a penalty:

```text
min || A Phi a - r ||^2   subject to   sigma_k . Phi (a_k - a_{k-1}) >= 0.
```

Eighty thousand linear inequalities on sixty-four unknowns, which an active-set
or cutting-plane loop handles because very few will be active. Then the question
has an answer: either a feasible point fits nearly as well, and the field can be
a plastic history, or the best feasible fit is far worse, and it cannot.

One caution for that run: the gradient of the penalty term was wrong here at
first — the adjoint of the increment `a_k - a_{k-1}` is a reverse difference
`q_j - q_{j+1}`, and a reverse cumulative sum was used instead. It left the
gradient `98 %` wrong and the optimiser motionless at every penalty, which read
exactly like "the constraint costs nothing". It is checked against finite
differences now, to `2.6e-9`.

## Retraction, and the test that deflates the rank-16 result

`scripts/qualify_subspace_prediction_and_dissipation_p43.py`, artefact
`subspace_prediction.json`.

**The 54.2 % figure is withdrawn.** It was computed with `sigma` built from the
strain *increment* since state 20, which makes it the stress increment
`d_sigma`, not the stress. State 20 carries load, so the absolute term is
missing and the sign of `sigma : d_eps_p` is not what was tested. Rebuilt with
`sigma_n = sigma_20 + C:(d_eps_sim - d_eps_p)`, the stress increment taken from
the **simulation** rather than the DIC, and the mid-point rule over all twenty
states instead of four long jumps:

| test | negative points |
|---|---:|
| the withdrawn one, `d_sigma : d_eps_p` | `0.526` |
| corrected, absolute stress, mid-point, 20 states | **`0.477`** |

Including `sigma_20` moves it from `0.526` to `0.477`, so the omission was real
but it was not hiding a sign flip. What the corrected number says is weaker and
different: at `47.7 %`, the reconstructed increments are **as often against the
stress as with it** — no preferential alignment, which is what an arbitrary
field gives, not evidence of inadmissibility. The earlier conclusion was both
wrongly computed and too strong.

One point in the other direction: the accumulated plastic path length, which is
the quantity irreversibility actually constrains, comes out at an RMS of
`6.20e-3` and a peak of `1.77e-2` — comparable to the archived accumulated
`5.67e-3`. The net-norm non-monotonicity flagged earlier was indeed not an
irreversibility violation, as the path length grows by construction.

## Leave one state out: the subspace does not predict

The rank-16 basis was built from the residuals of all four states, so
reproducing them is compression. Building it from three and asking for the
fourth separates compression from prediction:

| held out | rank 4 | rank 8 | rank 16 | rank 32 |
|---:|---:|---:|---:|---:|
| 25 | `0.633` | `0.579` | `0.544` | `0.515` |
| 30 | `0.421` | `0.371` | `0.344` | `0.331` |
| 35 | `0.372` | `0.317` | `0.288` | `0.264` |
| 40 | `0.599` | `0.571` | `0.521` | `0.496` |

**A basis that has not seen a state reproduces it only to between 26 % and
63 %**, and adding modes barely helps — from rank 4 to rank 32 the error moves
by a tenth. The missing content is not in the span the other states generate at
any rank.

So the rank-16 closure is **compression of four known fields, not a
low-dimensional mechanical subspace the material lives in**. The headline of the
previous section has to be read down accordingly: sixteen modes fit four states
because sixty-four coefficients fit four fields, not because the post-elastic
kinematics is sixteen-dimensional.

What survives is narrower and still worth having: a plastic eigenstrain of
realistic amplitude can reproduce each measured increment exactly, and the
accumulated path it implies is of the right size. What is not supported is that
these fields form a small shared subspace with predictive power, or that they
constitute a dissipative history.

### The path length is zig-zag, not accumulation

The agreement between the reconstructed path length, `6.20e-3`, and the archived
accumulated plastic strain, `5.67e-3`, is a coincidence of scale and should not
be read as support.

| quantity | value |
|---|---:|
| path length, RMS | `6.20e-3` |
| net displacement `\|eps_p(40)\|`, RMS | `1.38e-3` |
| ratio, RMS-weighted | **`4.48`** |
| ratio, median pointwise | `5.19` |

A monotone trajectory has a ratio of one. At `4.5` the reconstruction wanders
between four and five times further than it advances, which is exactly what a
dissipation split near `50/50` implies, and it is where the apparent agreement
with `5.67e-3` comes from. The net plastic strain is `1.38e-3`, matching the
state-40 amplitude found independently.

So the path-length check moves from the supporting column to the neutral one:
the *scale* of plastic activity is right, its *trajectory* is not a monotone
accumulation.

## The hard dissipation constraint: built, not solved

`scripts/qualify_hard_dissipation_constraint_p43.py`. **No result is quoted from
it**; the script is committed for the diagnosis it carries.

The formulation is the right one — minimise the misfit subject to
`D_k(q) >= 0` at every point and step, with the corrected mid-point dissipation
and the absolute stress, `sigma` refrozen between outer iterations because the
constraints are quadratic in the coefficients. Two solver failures stopped it.

`trust-constr` did not finish in ten minutes on 320 unknowns with a few thousand
constraints. Replacing it with an active set solved through a dense KKT system
is fast for small sets but cubic in the number of cuts, so admitting thousands
is slower again than what it replaced.

The second failure is the instructive one. An **add-only** active set is wrong as
soon as the cuts outnumber the unknowns: treating every active inequality as an
equality over-determines the system — a singular KKT at rank 8 with four states
— and in the limit forces the trivial solution. That is the same collapse the
penalty formulation produced, reached by a different route, and it would have
been easy to misread as "admissibility is unaffordable".

So the question stays open, and it is now precisely posed. Feasibility is never
in doubt: `a = 0` gives zero increments and zero dissipation. The entire content
is the **price**, and measuring it needs a genuine quadratic program —
multiplier-based dropping so the working set stays at most `n_unknowns`, or an
off-the-shelf QP solver. That is a bounded piece of work on a problem of 320
unknowns, and it is the last step before the Ludwik comparison can be
interpreted.

## Kelvin migration of the inverse core, and what it moved

`TensorPlasticObservabilityOperator` is now Kelvin internally. Engineering Voigt
survives only where an interface imposes it — `strain` returns it, and
`divergence_from_sample_stress` expects Voigt stress — with conversion at those
two calls and nowhere else.

Verified before reading anything into the replays:

| check | value |
|---|---:|
| `K = B_K^T C_K B_K` unchanged (the two scalings cancel) | `1.7e-15` |
| physical response of a given plastic tensor unchanged | `4.1e-16` |
| adjoint | `3.0e-16` |
| unit coordinate gives unit RMS `p_eq` | `1.000000` |

The gauge test was rewritten. It used to pin the inverse von Mises metric, which
was correct while the module stored engineering shear; it now asserts the
property that matters and is convention-independent — a unit coordinate vector
is a plastic field of unit RMS equivalent strain — so it survives the next
migration too.

### The physical conclusions are robust; the Krylov geometry improved

| quantity | engineering | Kelvin |
|---|---:|---:|
| residual at 128 LSQR iterations, state 40 | `0.0352` | `0.0405` |
| `p` RMS there | `7.97e-4` | `8.04e-4` |
| leave-one-out, state 35 held out, rank 32 | `0.264` | **`0.208`** |
| leave-one-out, state 40 held out, rank 32 | `0.496` | **`0.399`** |
| negative dissipation points | `0.477` | `0.452` |
| accumulated path RMS | `6.20e-3` | `6.28e-3` |

The trade-off, the amplitude and the path are unchanged to within a few per
cent: they are physical, and the convention does not touch them.

Prediction improved measurably. That is the expected direction: the engineering
norm counts shear twice, so it distorted the geometry in which QR, Krylov and
the truncated bases were built. The conclusion is unchanged in kind — a basis
built on three states still misses 20 to 40 % of the fourth, so it remains
compression rather than prediction — but it is less severe than the engineering
numbers suggested, and those numbers should not be quoted further.

## The dissipative QP runs; it is not yet converged

`scripts/solve_dissipative_plastic_history_p43.py`, with OSQP. The formulation
is the corrected one throughout: hard inequalities rather than a penalty, cuts
**rebuilt** at every outer iteration because a cut is only valid for the `sigma`
it was linearised at, mid-point dissipation with the absolute stress, and the
contraction as a plain Kelvin dot product.

One more trap found and fixed. Starting from `a = 0` is feasible, which makes it
a **fixed point of a cut-then-solve loop**: there is no violation to cut on, the
QP never runs, and the trajectory never leaves zero — four outer iterations
returned an error of exactly 1.000 with zero cuts. The solve has to come first,
with an empty constraint set, and the violations of *its* answer are what the
cuts are built from.

A reduced run — five states, rank 8, 1200 cuts — completes and gives:

| | mean error | negative points | path / net |
|---|---:|---:|---:|
| free | `0.328` | `0.433` | `4.48` |
| constrained | `0.957` | `0.448` | **`1.41`** |

**This is not the price of admissibility and must not be quoted as one.** With
1200 cuts against roughly two hundred thousand violated constraints, the
solution is being degraded by the cuts without reaching feasibility: the
negative fraction is unchanged at `0.45`. A number is only meaningful once the
violations are actually driven to zero.

What *is* meaningful is the structural effect. The path-to-net ratio falls from
`4.48` to `1.41` under partial constraint: the dissipation requirement removes
the zig-zag, which is exactly what it should do and what the free solution
lacked. The trajectory becomes nearly monotone before it becomes feasible.

Two things stand between this and the number. The cutting-plane needs to reach
feasibility — many more cuts per round, or a smarter selection than "the most
negative", since violations cluster and a spread-out set would constrain more
per row. And OSQP's cost grows with the constraint count, so the full run needs
either a tighter tolerance schedule or warm-starting between rounds, which the
solver supports and this script does not yet use.
