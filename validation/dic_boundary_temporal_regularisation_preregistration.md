# P43 measured-boundary temporal regularisation — preregistration

Date: 2026-07-30
Partition: P43, padded support `661 x 611` nodes, core bounds `[1440, 1800, 930, 1240]`.
Immutable input: `reference_data/dic_multistep_history_p0043_repaired_v1/repaired_history_mm.npy`.

## Why this campaign replaces Newton instrumentation

`dic_multistep_p0043_boundary_outlier_results.md` ruled out a *local* DIC
outlier at the two failing upper-boundary elements and concluded that the most
supported cause was nonlinear globalisation. That audit never tested the
**affine** component of the boundary motion as a function of time.

On the six archived states, the affine transverse strain increment is:

| state | `d eps_xx` | ratio to neighbour mean |
|---:|---:|---:|
| 1 | `-3.859e-05` | 1.04 |
| 2 | `-3.837e-05` | 1.04 |
| 3 | `-3.521e-05` | 0.95 |
| 4 | `-8.050e-05` | **2.18** |
| 5 | `-1.904e-05` | 0.52 |
| 6 | `-5.364e-05` | 1.45 |

State 4 is a `3.5 sigma` excursion immediately followed by a compensating
undershoot at state 5. A large increment followed by a small one is the
signature of a single displaced *state*, not of a physical loading event, and
state 4 is exactly where the measured-history solve fails.

Two further archived facts motivate a temporal rather than spatial treatment:

- boundary content at wavelength `<= 16 px` is `0.08 %` to `0.7 %` of the
  signal, and the non-affine residual is `0.0016--0.0033 px`, `26x` below the
  archived repeated-frame noise `sigma = 0.06283 px`. DISFlow has already
  removed the short-wavelength band (MTF-50 near `49 px`). A spatial filter
  would have almost nothing to remove, and Saint-Venant decay over `~38 px`
  against `150 px` of padding already protects the core from that band;
- the surviving noise is therefore long-wavelength and quasi-affine. A
  quasi-uniform boundary perturbation does not decay into the interior, so
  padding cannot suppress it. Its order of magnitude in affine strain is
  comparable to the physical per-state increment.

## Scientific position

The states are independent direct correlations of the same reference image
onto each deformed image. Measurement noise is therefore **independent in
time** while the physical boundary path is **smooth in time**. This asymmetry,
not any assumption about the material, is what licenses temporal
regularisation.

Every quantity below is labelled measured, computed or assumed:

- *measured*: the 41 displacement states, the raw images, `sigma`;
- *computed*: affine decompositions, singular spectra, noise estimates;
- *assumed*: that the physical boundary path is smooth on the sampling
  interval, and — only where stated — that the scalar loading parameter is
  monotone.

Monotonicity is a **closure hypothesis**, not a measurement. It is applied to
the scalar loading parameter only, never per node: a transverse node may move
non-monotonically under monotone remote loading.

## Stage 0 — measurement only, no mechanics

This preregistration covers stage 0 only. Stages 1--3 will be registered
separately once stage 0 has fixed the noise model.

Deliverable: `fem-inhouse diagnose-dic-boundary-loading-subspace`, writing
`reference_data/dic_boundary_loading_subspace_p0043_v1/`.

### S0.1 Temporal noise estimate

For every boundary degree of freedom, form the second temporal difference

`d2[k] = u[k+1] - 2 u[k] + u[k-1]`, `k = 1 .. 39`.

For independent noise of variance `sigma^2` on a locally straight path,
`Var(d2) = 6 sigma^2`. Report both the RMS estimator `RMS(d2)/sqrt(6)` and the
robust estimator `1.4826 * MAD(d2)/sqrt(6)`. Both are **upper bounds**: real
temporal curvature inflates them. The robust estimator is the registered one
because the campaign is about an outlier.

### S0.2 Empirical noise propagation

The `d2` fields divided by `sqrt(6)` are themselves realisations of the
measurement noise field **carrying its true spatial correlation**. Apply the
existing `affine_boundary_decomposition` to each of them and take the robust
spread of the resulting affine strains. This propagates noise to the affine
band without assuming spatial independence, which the archived `38.2 px`
autocorrelation forbids.

### S0.3 Loading subspace

SVD of the `41 x (boundary dof)` matrix, uncentred so that state 0 stays at the
origin. For each mode, report the singular value and the temporal roughness

`R = RMS(d2 of temporal coefficient) / (sqrt(6) * RMS(temporal coefficient))`.

`R` is near `1` for a temporally white mode and much smaller for a smooth one.
Registered separation rule: a mode is retained as signal when `R < 0.5`. The
mode count is therefore fixed by the temporal statistics, not by taste.

Registered falsifier: if no mode satisfies `R < 0.5`, or if the retained modes
carry less than `90 %` of the boundary displacement energy, the low-dimensional
loading model is refuted and the per-node route returns.

### S0.4 Outlier scoring over all 40 states

Robust z-score of the second temporal difference of the dominant loading
coefficient, and of the affine strain increments, over the full history rather
than against five neighbours.

Registered expectation, recorded before the full history is examined:
**state 4 must remain an outlier when scored against all 40 states**, at
`|z| >= 3` on at least one of the two scores. If it does not, the boundary
noise explanation of the state 3 to state 4 failure is withdrawn and Newton
globalisation returns as the leading hypothesis.

### S0.5 Per-state signal-to-noise

Report `|physical affine strain increment| / (sqrt(2) * sigma_affine)` for each
of the 40 increments, using the S0.2 noise. The `sqrt(2)` accounts for two
independent states entering one increment.

## What stage 0 must not do

- no mechanics, no PEEQ, no constitutive evaluation;
- no modification of the immutable history;
- no choice of smoother parameters. Stage 0 measures; stage 1 registers the
  smoother.

## Protocol boundary between design and test

Designing the smoother while looking at the 40 states is legitimate. The blind
part of this campaign is downstream and is registered here in advance:

1. smoother parameters frozen from stage 0 metrology, published before any
   mechanical run;
2. **Saint-Venant admissibility gate**: the regularisation applied to the
   already converged proportional baseline must leave the core, excluding the
   `150` element padding, unchanged within a tolerance registered in stage 1.
   If the archived converged result moves, the regularisation is inadmissible
   and everything downstream is void;
3. only then the measured-history mechanical replay.

Tuning the smoother until Newton converges would fabricate the result. It is
forbidden here for the same reason `Claude.md` forbids convergence-driven frame
rejection. `Claude.md` also forbids an *implicit* Kalman filter; the estimator
chosen in stage 1 will be declared explicitly with its noise model, which is
what that rule requires.

Registered failure outcome: if the regularised history still fails at the same
transition, the boundary-noise hypothesis is refuted and the negative result is
documented, not retried with a different smoother.
