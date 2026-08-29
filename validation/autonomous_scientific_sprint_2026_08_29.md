# Autonomous scientific sprint — 2026-08-29

This sprint used archived repeated-frame noise, displacement fields and
Jacobians only. No mechanical forward, finite-difference sensitivity,
optimization, FEMU run, or new image-level DIC calculation was launched.

## 1. Commits produced

| SHA | phase | result |
|---|---|---|
| `d0d55313` | structured-noise decomposition | translation/affine energy audit |
| `d783010d` | nuisance projection | N0/N1/N2 modal calibration |
| `da49058d` | spatial cross-validation | 30 non-overlapping field tiles |
| `88d8c3b1` | temporal information | per-state, cumulative and subset SVD |
| `f7cf113d` | prior/final nonlinearity | frame-resolved local linearity check |
| `2719fb4b` | REGM projection gate | blocked: scalar objectives only |
| `799db54d`, `3fb81303` | latent-state atlas | inverse/observability aggregation |
| `af55596a` | morphology gate | blocked: no common J2/SRIX/DIC support |

## 2. Noise model

The hydrated payload is `(3600, 3100, 2)` `float32`.

- Structured bias: **yes**. Affine removal removes 96.6% of full-field energy,
  98.3% in the P43 core, 98.3% in the solve neighbourhood, and 99.3% in the
  calibration corner. Translation alone removes 88.2% in the core.
- The spatial cross-validation is not a single isolated anomaly: tile RMS
  ranges from `1.79e-5` to `1.51e-4` mm for `u_x` and `2.14e-5` to `1.89e-4`
  mm for `u_y`, with affine-energy fractions `0.869–0.998`.
- N1 translation projection strongly reduces modal means but leaves standard
  deviations around `0.48–0.85` in the P43 zones.
- N2 affine projection centres the modes (`|mean| < 0.02`) but collapses their
  dispersion to roughly `0.07–0.12`.
- Therefore nuisance projection is not yet a calibrated absolute likelihood;
  the registered corner whitener remains unchanged.

**Noise conclusion:** structured bias is present, but no tested nuisance
projection simultaneously provides zero bias and unit-scale modal variance.

## 3. Temporal information

The archived final M20 wrap-free surrogate has singular values
`(0.33607, 0.05106, 0.01077, 8.78e-6)` and normalized spectrum
`(1, 0.15193, 0.03205, 2.61e-5)`.

- The first two directions dominate early and transition states.
- The `Q+b`-like third direction grows mainly in the later states; the fourth
  remains effectively null.
- The one-based frame subset `{6, 7, 8}` retains about 98.1% of the full Fisher
  trace and has a rank-3 angle of about `0.001°` to the full set. This is a
  geometric subset result, not a temporal statistical optimum.
- Prior-to-final frame checks show strong local-linearity error (relative
  errors about `1.3–5.0`); late-frame correlation improves, reaching about
  `0.95` for the final state. Rank-3 subspaces remain stable while individual
  leading directions rotate substantially.

## 4. REGM

Compatible observable-vector projection: **no**. The registered REGM reports
contain scalar objectives and timings, not residual vectors on the same
eight-state `21x21x2` support. No interpolation or support-forcing was done.
The exact-space ranking remains supported (`Spearman 0.866`), while the
observed transfer ranking remains a separate negative result (`Spearman
0.326` at T1).

## 5. Latent-state atlas

The aggregated artifacts support three strong conclusions:

1. A free tensor inverse can fit the observable while retaining a large latent
   nullspace (19 null directions; best registered gauge error about 0.797).
2. A constrained `q=1` local representation can recover an exact synthetic twin
   (`2.13e-4` field error, condition number about 200).
3. Enrichment (`q=4`) introduces basis null directions (23 of 144), and FCC
   tensor consistency does not establish individual slip recovery (unconstrained
   `e_FCC` near `1.8e-12`, weighted system `R²` about `-0.031`).

These cases are not directly comparable likelihoods; together they establish
that observable agreement is not latent-state validation.

## 6. Morphology

Blocked conservatively. The archived J2, SRIX and DIC fields do not share a
declared common crop, increment, support and observation representation. No
multiscale comparison was manufactured.

## 7. FCC slip observability

No new Schmid-tensor SVD was run. Existing FCC decomposition artifacts were
used in the atlas: tensor reconstruction can be essentially exact while the
individual slip-system recovery gate remains false.

## 8. Recommendation

**A — improve/qualify the observation-noise model.**

The repeat-frame contains a large spatially structured component. Translation
and affine projection remove its modal bias, but the resulting variance is not
calibrated; a local stationary whitener also worsens modal calibration. The
temporal SVD identifies a useful geometric rank-3 space, but not an absolute
experimental SNR. Therefore a boundary-only FEMU should not be started yet.
The next defensible step is to obtain a declared observation-noise model with
independent spatial/temporal validation, then revisit detectability before any
expensive identification.
