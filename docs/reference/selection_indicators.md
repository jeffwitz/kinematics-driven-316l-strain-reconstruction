# Fluctuation defects and the selection rule

**Category: Reference.** Exact definitions for the four defects of the P43
`(ell, alpha)` selection campaign, their normalisation, the spatial bootstrap
and the decision rule applied to them. It documents what the software computes,
not what any campaign concluded.

Protocol: `validation/p0043_small_parameter_matrix_preregistration.md`.
Results, when they exist: `validation/p0043_indicator_validation_results.md`.

## The observable

Every defect reads one field, the high-passed strain magnitude:

```
eps   = sym(grad u)                       small strain of the observed displacement
H_s   = f - G_s * f                       declared high pass, Gaussian sigma = s / 2
g_s   = || H_s(eps) ||_F                  scalar fluctuation field
```

The displacement is always the **observed** one: warped onto the reference
image and re-measured through DISFlow. No primary defect compares the DIC to a
raw FEM field.

Differentiation, support, mask and edge handling are the historical EVM
operator's: `np.gradient` with array axis 0 = canonical x, then `cell_average`
to element centres, differentiated on the solve grid and cropped to the core
afterwards so no core point uses a one-sided difference.

**Principal scale `49 px`**, the measured MTF-50 of the chain. `32` and `96 px`
are sensitivities. `8` and `16 px` are excluded from selection: the coupled
fields carry `10` to `18 %` of the DIC high-pass energy there and the replay
adds no speckle-decorrelation noise, so those scales measure missing
measurement noise as much as missing model content.

## The four defects

All lower-is-better, all computed on `g_49`.

| Symbol | Definition | What it catches |
|---|---|---|
| `D_shape` | `1 - pearson(g_FEM, g_DIC)` | are the fluctuations in the same places |
| `D_amplitude` | `abs(log(q95_FEM / q95_DIC))` | is the upper tail the right size |
| `D_localisation` | `1 - FSS` at the DIC's q90 threshold, neighbourhood `49 px` | multiscale agreement of the active set |
| `D_presence` | `abs(log R)`, `R = ||H(eps_FEM)||^2 / ||H(eps_DIC)||^2` | is there any fluctuation at all |

Two properties are deliberate.

**The logarithms make over- and under-shoot cost the same.** Halving the
amplitude and doubling it give the same defect, which a plain ratio would not.

**`D_localisation` freezes its threshold on the reference.** The q90 comes from
the DIC and is applied unchanged to the candidate, so no candidate can move the
boundary it is judged against. A rescaled candidate therefore *does* change its
own active set; that is correct, not an invariance failure.

### Why presence is separate

`D_presence` exists because a fluctuation distance cannot reject a smooth
field. A relative `L2` distance saturates near `1` for any candidate with no
content at the scale considered, since the residual reduces to the reference
itself: predicting nothing costs `~1`, predicting something slightly misplaced
costs more than `1`. The earlier gradient diagnostic measured exactly that — a
structureless control ranked third of six.

Presence is measured separately and **never merged** into the others. It enters
the Pareto front as a fourth coordinate and the minimax as a fourth term; no
weighted sum is formed anywhere.

## Normalisation

```
Z_k = (D_k - D_k_self) / (D_k_null - D_k_self)
```

- `D_self` is the **measurement floor**: the DIC perturbed by a synthetic
  repetition residual, ten realisations, median. The residual keeps the measured
  `38.2 px` coherence and is scaled to the measured spurious EVM RMS,
  `1.363e-4`, **not** to the measured displacement deviations. Those two
  archived quantities disagree by a factor twelve, because the real residual is
  far smoother than a Gaussian field of that nominal coherence; the indicators
  consume strain, so the strain is the anchor.
- `D_null` is the **best score either negative control reaches**, declared per
  indicator, with the control recorded. The two controls fail in different
  ways and neither is uniformly the harder bar.

`D_self` is an upper bound on the floor, since the archived repeated pair is
itself a noise-and-drift upper bound. Every `Z` is therefore a lower bound.

The normalisation is anchored on constructed controls, which is a known
weakness: each indicator is divided by a different arbitrary number and the
minimax then compares those quotients. It is bounded three ways — the control
is declared per indicator, the Pareto front is computed on the **raw** defects
where the normalisation cannot act, and the selection is repeated with the
other control as anchor.

## Decision rule

**Front.** Pareto domination on the four raw defects. Domination is invariant
under any per-indicator monotone rescaling, so the front is identical on `Z`;
computing it on raw values makes that invariance visible rather than implicit.

**Tie-break.** `J_inf = max(Z_shape, Z_amplitude, Z_localisation, Z_presence)`.
The retained candidate minimises its worst normalised defect. A minimax, not a
sum, so an excellent amplitude cannot buy a very bad localisation.

**Stability.** Paired spatial bootstrap, `10 000` draws, seed `20260801`,
square tiles of `49 px` giving 42 whole tiles on the `360 x 310` core. Tiles
`32` and `96 px` are sensitivities.

Tiles are `49 px` rather than the 8-unit blocks of the earlier section
bootstrap: that resampling was one-dimensional along a centreline, this one is
two-dimensional, and an `8 px` tile sits far below the measured coherence, so it
would treat correlated pixels as independent and understate the uncertainty.

Every candidate is scored on the **same** tile multiset in a given draw. An
unpaired comparison would mix model difference with resampling difference.

**Zone.** A candidate is indistinguishable from the best when the bootstrap
interval of the **paired difference** `J_inf(candidate) - J_inf(best)` contains
zero. Comparing marginal bands is the interval-overlap fallacy: the draws share
their tiles, so the difference has far less spread than either score alone.

The zone is always reported as an explicit point list, never as a range of
`ell` crossed with a range of `alpha` — a non-dominated set on a grid need not
be a rectangle, and a bounding box would claim points that were never
preferred.

**Solver floor.** One grid point recomputed at 40 increments instead of 20. The
bootstrap resamples the observation, not the solve, so it cannot say how much
of a difference between neighbouring points is the solver recomputing the same
physics. Two points closer than this floor are indistinguishable whatever the
bootstrap says.

## Indicator validation

No defect enters a decision until it has been applied to nine cases whose answer
is known: the DIC against itself, the repetition residual, both negative
controls, amplitude at `0.80` and `1.20`, a band displaced by `16 px`, a band
removed, bands merged, and a spurious band.

An indicator is accepted only if identity is optimal, the homogeneous control
fails on presence and amplitude, the translated control fails on shape or
localisation, amplitude and position errors are not mistaken for one another, a
removed band is worse than a moderate amplitude error, and the verdicts agree
at `32`, `49` and `96 px`.

**An indicator that fails stays in the report as a diagnostic and leaves the
selection.** The band-level cases act through a hard-edged corridor whose
boundary step inflates the high-pass energy by two orders of magnitude, so only
their sign is read, never their magnitude.

## What no result from this tooling licenses

A provisional reconstruction parameterisation for one ROI. Not an internal
length of 316L, not transferability, and neither a validation nor a refutation
of the nonlocal formulation.
