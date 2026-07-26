# Joint identification of spatial length and coupling modulus

## What this experiment is allowed to conclude

The micromorphic J2 model contains two parameters:

$$
R(p,\chi)=R_{\mathrm{local}}(p)+H_\chi(p-\chi),
\qquad
\chi-\ell^2\Delta\chi=p.
$$

The purpose of this workflow is to determine whether the DIC data can
distinguish the coupling modulus \(H_\chi\) from the spatial length \(\ell\).
It is not an identification of an intrinsic material length. Such a claim
would additionally require unchanged-parameter transfer to another
band-containing ROI, another observation resolution and, ideally, another
loading history.

This distinction controls the whole design:

- **F0** is a frozen-field, DCT-only heuristic;
- **F1** is reduced but fully coupled mechanics used only for ranking;
- **F2** is the unchanged full-resolution calculation used for scientific
  conclusions;
- no F2 calculation is launched by the identification workflow;
- the local case \(\alpha=0\) is stored once because \(\ell\) has no meaning
  when \(H_\chi=0\).

Here

$$
\alpha=\frac{H_\chi}{H_{\mathrm{ref}}},
\qquad
A_\chi=H_\chi\ell^2.
$$

\(H_{\mathrm{ref}}\) is always read from the campaign metadata. It is never
hard-coded in the workflow.

## Why a full two-dimensional grid would be wasteful

For a Fourier mode of wavenumber \(k\), the Helmholtz equation gives

$$
\widehat\chi(k)=\frac{\widehat p(k)}{1+\ell^2 k^2},
$$

and therefore

$$
H_\chi\bigl(\widehat p-\widehat\chi\bigr)
=H_\chi\frac{\ell^2k^2}{1+\ell^2k^2}\widehat p.
$$

When \(\ell k\ll1\), the response depends primarily on
\(H_\chi\ell^2=A_\chi\). Different \((H_\chi,\ell)\) pairs can then be almost
indistinguishable. A dense full-resolution grid would repeatedly solve
expensive mechanical problems along this near-degenerate direction.

For this reason the workflow records and visualizes both parameter systems:

$$
(\ell,\alpha)
\quad\text{and}\quad
(\theta_H,\theta_A)
=\left(\log H_\chi,\log A_\chi\right).
$$

```{image} ../_static/joint_identification/joint_identification_parameter_maps.*
:alt: Sparse design in physical and identifiability coordinates.
:align: center
:width: 100%
```

The right-hand panel is not cosmetic. A narrow cost valley parallel to one of
these log-coordinate directions is direct evidence that the two parameters
are poorly separated by the present observation.

## Observation and objective contract

The primary observable is the historical total equivalent strain
reconstructed from displacement. DIC and FEM pass through the same recorded
operator \(\mathcal M_{\mathrm{DIC}}\), which fixes:

- the displacement unit and strain convention;
- the grid transformation;
- the finite-difference strain operator;
- any spatial averaging;
- the finite-value mask;
- the retained partition core.

Padding is excluded from physical metrics. `PEEQ` remains an internal
mechanical diagnostic and is never presented as an experimental field.

The workflow deliberately keeps amplitude and localization separate.

### Amplitude objective

For \(q\in\{50,75,90,95,99\}\),

$$
e_q=\log\frac{Q_q(E_{\mathrm{FEM}})}{Q_q(E_{\mathrm{DIC}})},
$$

and

$$
J_{\mathrm{amp}}
=\sum_q w_q e_q^2
+w_\sigma
\left[
\log\frac{\sigma_{\mathrm{FEM}}}{\sigma_{\mathrm{DIC}}}
\right]^2.
$$

The implementation uses a documented positive floating-point floor only when
a logarithm would otherwise receive zero. It does not normalize either field.

### Localization objective

Two overlap families are retained:

- top-10% FEM against top-10% DIC, which tests relative position;
- one absolute threshold equal to DIC \(Q_{90}\), applied numerically to both
  fields, which also detects peak suppression.

The Pareto objective used here is

$$
J_{\mathrm{loc}}=1-\mathrm{IoU}_{\mathrm{DIC}\ q90}.
$$

RMSE, relative L2, Pearson and Spearman correlations, gradient RMS, total
variation, quantile errors and radial spectra remain visible alongside the
two selection objectives.

## F0 — one DCT per length, not per parameter pair

F0 starts from the converged local PEEQ field \(p_0\). For every length:

$$
\chi_\ell-\ell^2\Delta\chi_\ell=p_0,
\qquad
r_\ell=p_0-\chi_\ell.
$$

On the structured element-centre grid, homogeneous Neumann conditions are
diagonalized by an orthonormal discrete cosine transform. The eigenvalues are

$$
\lambda^x_i=\frac{2-2\cos(\pi i/n_x)}{h_x^2},
\qquad
\lambda^y_j=\frac{2-2\cos(\pi j/n_y)}{h_y^2},
$$

so the solve is

$$
\widehat\chi_{ij}
=\frac{\widehat p_{0,ij}}
{1+\ell^2(\lambda^x_i+\lambda^y_j)}.
$$

This gives several concrete accelerations:

1. no sparse Helmholtz matrix is assembled or factorized;
2. one forward and one inverse DCT solve the complete padded field in
   \(O(N\log N)\);
3. \(r_\ell\) depends on \(\ell\), but not on \(H_\chi\);
4. every \(\alpha\) at the same length reuses the same DCT result through
   \(H_\chi r_\ell\);
5. local and gradient energies are rescaled without another solve.

The P43 screen used 22 lengths and 21 positive alpha levels. Including the
single local point, it evaluated 463 couples with only 22 DCT solves in
**7.84 s**, with a measured peak RSS of **141,404 KiB**.

This is the main reason F0 can be dense. An equivalent F2 grid would require
hundreds of full nonlinear finite-element calculations.

### What F0 predicted — and what it did not

At \(\ell=58.88\,\mu\mathrm m\), the F0 proxy strength was compared with the
existing F2 calculations at \(\alpha=0,1,2,4\). Its rank correlation with F2
relative L2 and with the PEEQ peak, standard deviation and total variation
was \(-1\): it recovered the observed monotone ordering.

That result validates F0 as a **screening direction**, not as a response
surface. Plasticity is not reintegrated, equilibrium is not recomputed and
the final EVM cannot be predicted from \(H_\chi r_\ell\). No F0 value is used
as a scientific FEM-DIC error.

## F1 — reduced coupled mechanics

F1 is not a second finite-element implementation. It prepares a physically
coextensive reduced dataset and delegates the calculation to the production
`PartitionWorkflow`.

For the factor-two P43 model:

| Property | F2 | F1 |
|---|---:|---:|
| global element grid | 3600 × 3100 | 1800 × 1550 |
| padded P43 solve | 660 × 610 | 330 × 305 |
| retained core | 360 × 310 | 180 × 155 |
| padding | 150 elements | 75 elements |
| spacing | 1.84 µm | 3.68 µm |
| loading increments | 20 | 10 |

Element material fields are area-averaged. DIC displacement boundary values
are sampled only at physically coincident coarse nodes. Physical dimensions,
the core/padding separation, material units and the complete loading history
are preserved. Every candidate starts from the initial constitutive state;
final internal variables are never transferred from another parameter pair.

The gate

$$
\ell/h_{\mathrm{F1}}\ge 3
$$

prevents an unresolved nonlocal length. It equals 16 at
\(\ell=58.88\,\mu\mathrm m\), and still equals 5.43 at the shortest tested
length of \(20\,\mu\mathrm m\).

### Coupled-solver acceleration already present underneath F1 and F2

F1 benefits from the same verified production optimizations as F2:

1. intermediate micromorphic fixed-point calls ask MFront only for the
   updated PEEQ and do not compute a tangent or full 3D tensors;
2. one MFront call with the consistent tangent follows fixed-point
   convergence and supplies the mechanical Newton assembly;
3. full 3D tensor completion occurs only for the accepted final state;
4. Kelvin, PEEQ and nonlocal-field buffers are preallocated and reused;
5. the proportional DIC predictor direction is assembled once;
6. the free-free CSR graph is fixed and only its numerical values change;
7. PARDISO phase 11 is performed once, followed by explicit phase 22/33
   pairs;
8. verified J2 tangents use the upper-triangular CSR graph and symmetric
   positive-definite `mtype=2`;
9. unknown future behaviours, including crystal plasticity by default, retain
   the complete graph and general `mtype=11`.

These changes do not modify the constitutive model, \(\ell\), \(H_\chi\),
fixed-point tolerance, Newton method, tangent definition or cutback policy.
On the complete P187 gate, fixed CSR reduced the process time from
273.56 s to 244.67 s; the subsequent `mtype=2` gate reduced it to 227.34 s.
The detailed implementation and gates are described in
{doc}`micromorphic_plasticity`.

### F1 validation against existing F2

F1 was validated before it was allowed to rank new candidates:

| α | F1 wall time (s) | F1 correlation | F2 correlation | F1 relative L2 | F2 relative L2 |
|---:|---:|---:|---:|---:|---:|
| 0 | 127.06 | 0.4033 | 0.3791 | 0.8922 | 0.9516 |
| 1 | 217.32 | 0.4895 | 0.4624 | 0.5954 | 0.6174 |
| 2 | 243.72 | 0.5090 | 0.4814 | 0.5071 | 0.5256 |
| 4 | 289.60 | 0.5319 | 0.5036 | 0.4177 | 0.4341 |

The four-point validation took 883.21 s wall time and peaked at
1,428,156 KiB RSS. It passed all pre-declared gates:

- identical L2 ranking;
- identical correlation ranking;
- maximum absolute correlation error below 0.05;
- relative L2 error below 15%;
- top-10 and absolute-q90 IoU errors below 0.05.

F1 is therefore authorized to **rank** points. It is not a substitute for F2
and its fields are not reported as final scientific predictions.

## The sequential design, including failures

The initial sparse design was the Cartesian support

$$
\ell\in\{20,40,60\}\,\mu\mathrm m,
\qquad
\alpha\in\{1,3.5,6\}.
$$

This is nine coupled F1 calculations, not a fine grid. Seven converged.
The points \((20\,\mu\mathrm m,3.5)\) and
\((20\,\mu\mathrm m,6)\) reached the existing minimum cutback and failed
cleanly. Their partial constitutive states were not reused.

The correct DOE response was not to change Newton, tolerances or the material
law. Two targeted points, \(\alpha=2\) and 2.5 at 20 µm, were added to bracket
the highest converged short-length coupling. Both converged. This is a
sequential design decision: information from failed points changes where the
next observations are placed, while the numerical and physical model remains
frozen.

The initial nine-point attempt took 2,967.86 s and peaked at 1,443,392 KiB.
The two adaptive points took 630.0 s and peaked at 1,403,376 KiB. Every point
has an individual immutable manifest and status; failure is data, not a
silently missing row.

### Instrumented diagnosis of the short-length failure

The original failure message identified only the global symptom: the
increment was repeatedly cut back below the minimum. A cache-isolated replay
of \((20\,\mu\mathrm m,3.5)\) retained all historical F1 controls and added
iteration tracing.

The replay reached seven converged increments before the first failure:

| Diagnostic | Observed value |
|---|---:|
| first failed pseudo-time | 0.8 |
| Newton limit | 15 |
| relative Newton residual at iteration 15 | \(3.2091\times10^{-6}\) |
| configured Newton tolerance | \(3.0\times10^{-6}\) |
| maximum micromorphic iterations in any call | 12 |
| failed micromorphic calls | 0 |
| minimum yield-surface radius | 31.597 MPa |
| largest absolute \(H_\chi(p-\chi)\) | 134.192 MPa |
| largest Helmholtz residual | numerical roundoff scale |

The fixed point therefore **was not the source of these cutbacks**. At
the first failed increment, every micromorphic call converged. Its iteration
count fell from 11--12 in the first two Newton trials to one near the
mechanical limit. The mechanical residual decreased monotonically but reached
the maximum Newton iteration just above its tolerance. Eleven cutbacks then
reproduced the same mechanical-limit failure; the final attempted step was
\(9.765625\times10^{-5}\).

This evidence changes the numerical response:

- the parameter pair is not rejected by the yield-radius positivity check;
- increasing the micromorphic iteration budget cannot solve the observed
  failure;
- Aitken is available and tested, but is not activated merely because the
  global solve failed;
- the causal follow-up changes only the Newton iteration ceiling from 15 to
  25, with increments, tolerances, Picard relaxation and cutback policy held
  fixed.

That causal follow-up converged:

| Diagnostic | Newton-25 replay |
|---|---:|
| converged increments | 10 / 10 |
| cutbacks | 0 |
| total Newton iterations | 134 |
| maximum Newton iterations | 17 |
| wall time | 441.85 s |
| total micromorphic iterations | 546 |
| maximum micromorphic iterations | 12 |
| minimum yield-surface radius | 28.386 MPa |
| residual-direction cosine range | 0.999897 to 0.999996 |

The positive residual-direction cosines rule out the oscillatory signature
that would justify Aitken for this replay. The point also extends the
short-length F1 trend beyond \(\alpha=2.5\): relative L2 decreases from
0.5707 to 0.5358, correlation rises from 0.5099 to 0.5199, and the amplitude
objective decreases from 0.8191 to 0.6491.

The same Newton-25 profile was then applied to \(\alpha=6\), only after the
3.5 result had converged:

| Diagnostic | \(\alpha=6,\ell=20\,\mu\mathrm m\) |
|---|---:|
| converged increments / cutbacks | 10 / 0 |
| wall time | 503.21 s |
| total / maximum Newton iterations | 174 / 22 |
| total / maximum micromorphic iterations | 646 / 12 |
| minimum yield-surface radius | 33.248 MPa |
| relative L2 / correlation | 0.4787 / 0.5367 |
| amplitude objective | 0.4007 |
| top-10 / absolute-q90 IoU | 0.2619 / 0.2660 |

The short-length amplitude profile therefore remains monotone through
\(\alpha=6\); the former \(\alpha=2.5\) boundary was numerical, not a
scientifically identified optimum. The already-generated F2 proposal that
used \((20\,\mu\mathrm m,2.5)\) is consequently **superseded and must not be
launched**. A new immutable F1 collection and F2 proposal are required before
production calculations, with the Newton-25 policy declared consistently.

The likely architectural limitation is the staggered mechanical tangent: it
is consistent for the local material update at fixed \(\chi\), but does not
contain the complete derivative of the converged Helmholtz coupling. At strong
short-length coupling this can reduce global Newton to slow linear
convergence. A monolithic block Newton or a consistently coupled tangent
remains the long-term remedy; neither is introduced by the diagnostic replay.

The raw trace is stored outside the identification cache under
`results/joint-nonlocal-fixed-point-diagnostic-p0043`. The compact
reproducible configuration is
`configs/joint_nonlocal_fixed_point_diagnostic_p0043.yaml`. The isolated
Newton-25 confirmation uses
`configs/joint_nonlocal_newton_diagnostic_p0043.yaml`.
The compact evidence record is
`validation/joint_nonlocal_fixed_point_diagnostic_p0043.json`.

## Profiles and Pareto front

At each sampled length, a monotone PCHIP is fitted only inside the converged
alpha interval. No monotonicity is assumed before checking the samples.

| ell (µm) | converged α samples | profiled α* | status |
|---:|---|---:|---|
| 20 | 1, 2, 2.5, 3.5†, 6† | 6.00 | upper sampled boundary |
| 40 | 1, 3.5, 6 | 6.00 | upper sampled boundary |
| 60 | 1, 3.5, 6 | 6.00 | upper sampled boundary |

† The two added short-length points use the separately declared Newton-25
diagnostic policy. They supersede the old numerical boundary but have not yet
been folded into a regenerated immutable collection.

```{image} ../_static/joint_identification/joint_identification_h_profiles.*
:alt: Profiles of the amplitude objective at each sampled length.
:align: center
:width: 90%
```

Every amplitude profile is still decreasing at its largest converged
coupling. Consequently:

- there is no interior amplitude optimum;
- \(H_\chi\) and \(\ell\) are not yet separately identified;
- extrapolating the PCHIP outside its sampled interval is forbidden;
- a quadratic fit in \((\log H_\chi,\log A_\chi)\) is stored only as a
  conditioning diagnostic, not as an uncertainty estimate.

The sparse quadratic diagnostic is numerically well posed on the sampled
points: the normalized-coordinate Hessian condition numbers are 4.85 for
amplitude and 3.34 for localization. The corresponding local parameter
correlations are 0.657 and 0.494, with fitted \(R^2\) values of 0.998 and
0.915. These numbers show that the sampled surface is smooth enough to
describe locally. They do **not** establish identifiability: the direct
profiles still place every amplitude minimum on a boundary, which is the
stronger and more transparent result.

The non-dominated front keeps amplitude and localization visible:

```{image} ../_static/joint_identification/joint_identification_pareto.*
:alt: Amplitude-localization Pareto front for F1 and reused F2 points.
:align: center
:width: 90%
```

The current F1 front contains:

- 60 µm, α=6: lowest \(J_{\mathrm{amp}}\);
- 58.88 µm, α=4: best absolute-q90 localization, already available in F2;
- 40 µm, α=6: normalized Pareto knee.

The fact that different points optimize amplitude and localization is exactly
why no a-posteriori weighted scalar objective is introduced.

## Homogeneous experiment for separating strength and length

The profiles above mix the historical Newton-15 collection with two
cache-isolated Newton-25 diagnostic replays. They are enough to disprove the
old short-length boundary, but they are not a valid dataset for comparing
parameters quantitatively. The replacement experiment therefore replays
**every** F1 point with one immutable numerical policy:

- 25 mechanical Newton iterations maximum;
- 10 imposed loading increments;
- relative Newton tolerance \(3\times10^{-6}\);
- fixed Picard relaxation 0.5;
- 15 micromorphic iterations maximum;
- no opportunistic Aitken acceleration;
- identical cutback policy and factor-two spatial reduction.

The versioned configuration is
`configs/joint_nonlocal_identifiability_p0043_newton25.yaml`. It produces 23
unique F1 calculations including the single local control. A point may carry
several experimental roles, but it is solved only once.

### Experiment 1 — find a plateau in coupling strength

At each of \(\ell=20,40,60\,\mu\mathrm m\), the profile contains

$$
\alpha=6,\ 9,\ 12.
$$

The purpose is not to force a minimum. An asymptotic plateau is sufficient to
bound the identifiable coupling strength. Between successive levels the
workflow records:

- relative decrease of \(J_{\mathrm{amp}}\);
- Pearson-correlation gain;
- relative change of L2 error;
- degradation of the apparent band-width error;
- PEEQ peak and spatial redistribution.

The pre-registered plateau thresholds are respectively 3%, 0.005, 2% and 5%.
They are stored in the configuration, not selected after seeing the fields.
If any length still improves at \(\alpha=12\), the workflow reports
`needs_alpha_saturation` and refuses to generate an F2 proposal.

This distinction matters scientifically. A best result at the last sampled
alpha is a lower bound on useful coupling, not an identified modulus. If the
improvement remains strong through 12, \(H_\chi\) may be compensating for a
deficiency of the local J2 model rather than representing an independently
identified physical parameter.

### Experiment 2 — hold \(A_\chi\) constant

The decisive degeneracy test uses the anchor

$$
(\ell,\alpha)=(20\,\mu\mathrm m,6)
$$

and the two equal-\(A_\chi\) points

$$
(30\,\mu\mathrm m,2.666666\ldots),
\qquad
(40\,\mu\mathrm m,1.5).
$$

Because the same \(H_{\mathrm{ref}}\) applies to all three,
\(\alpha\ell^2\), and therefore \(A_\chi\), is identical. The implementation
checks the numerical spread of \(A_\chi\) before interpreting the response.

If EVM, band width, band orientation, correlation lengths and spectra remain
indistinguishable, the present experiment identifies only
\(A_\chi=H_\chi\ell^2\). If amplitudes remain similar but spatial scales
differ beyond numerical, mesh, DIC-resolution and between-ROI variability,
then \(\ell\) carries independent observable information. F1 records these
differences; it does not by itself establish the required uncertainty
thresholds.

### Experiment 3 — hold alpha constant

The orthogonal check compares

$$
\alpha=6,
\qquad
\ell=20,\ 40,\ 60\,\mu\mathrm m.
$$

At fixed coupling strength, an observable variation of apparent band width,
axis position, autocorrelation length or directional spectral content is the
signature expected from a spatial length. Global L2 alone is insufficient:
it can prefer a field with the right amplitude but the wrong morphology.

### Three distinct metric families

The redesigned analysis keeps three families separate:

| Family | Quantities | Main sensitivity |
|---|---|---|
| amplitude | EVM quantiles 50/75/90/95/99, standard deviation, RMSE, L2 | mostly \(H_\chi\) |
| localization | relative top-10 IoU, absolute DIC q80/q90/q95 overlap, active area, band-axis offset | position and support |
| spatial scale | band width and extent, orientation, x/y correlation lengths, gradient RMS, total variation, spectral centroid and radial-spectrum distance | mostly \(\ell\) |

The same absolute DIC threshold is applied to DIC and FEM for the absolute
overlap and band measurements. This prevents peak suppression from appearing
as a localization improvement merely because each field selects its own top
fraction.

### Use the loading history, not only the final image

Every F1 point stores converged `U`, `E` and `PEEQ` snapshots at 25%, 50%, 75%
and 100% of the proportional loading history. The DIC reference at a fraction
is reconstructed by scaling the imposed displacement, then passing it through
the same observation operator as the FEM snapshot. No final constitutive
state is transferred between parameter points.

The history tests whether one length describes band initiation, broadening,
merging and propagation consistently. A strength parameter can alter the
rate at which amplitudes grow while a spatial length should leave a more
persistent signature on morphology. This extra axis of observation is
essential because a single final image can hide that distinction.

### Explicit decision gate before F2

The workflow cannot write a new high-fidelity manifest until:

1. all homogeneous saturation profiles are complete;
2. all constant-\(A_\chi\) points are complete;
3. all fixed-alpha length points are complete;
4. every alpha profile reaches the pre-registered plateau.

Even after those gates pass, at most three F2 points may be proposed:

1. an amplitude candidate on the plateau;
2. a different-length candidate testing the same \(A_\chi\);
3. the candidate best reproducing localization or band scale.

The three points must discriminate hypotheses; they must not be neighbouring
samples from the same Pareto branch. Their commands remain inert until
explicit human approval. Transfer to a second band-containing ROI uses the
same parameters and observation operator without recalibration.

The possible scientific outcomes are deliberately limited:

- \(H_\chi\) and \(\ell\) are separately identifiable;
- only \(A_\chi\) is identifiable, in which case any reported effective
  length depends on a stated convention;
- alpha does not saturate, suggesting that coupling is acting as an amplitude
  corrector or that the current data/model do not identify it.

None of these F1 outcomes authorizes the phrase *material internal length*.

## Why the staged calculation is faster

```{image} ../_static/joint_identification/joint_identification_cost_hierarchy.*
:alt: Measured cost hierarchy of F0, F1 and F2.
:align: center
:width: 90%
```

The bars compare the dense F0 screen with a median converged F1 point and a
median positive F2 point on the same workstation:

- F0: 7.84 s for 463 frozen-field pairs;
- F1: 271.4 s for one reduced coupled point;
- F2: 1,792.4 s for one full-resolution coupled point.

The levels answer different questions, so this is not a claim that F0 is
228 times faster for the *same result*. F0 cheaply rejects or groups
parameters. F1 pays for coupled mechanics to validate the ranking. F2 is paid
only for a small, justified set of scientific points.

## Historical full-resolution proposal — superseded, not launched

The following manifest is retained for provenance only. It predates the
Newton-25 diagnosis and **must not be executed**.

The generated immutable manifest proposes four, not five, new F2 runs:

| ell (µm) | α | Hχ (MPa) | Aχ (MPa mm²) | estimated time | purpose |
|---:|---:|---:|---:|---:|---|
| 58.88 | 6 | 31,008.885 | 107.5033 | 0.710 h | close the existing α boundary |
| 60 | 3.5 | 18,088.517 | 65.1187 | 0.577 h | best localization under amplitude constraint |
| 40 | 6 | 31,008.885 | 49.6142 | 0.710 h | Pareto knee |
| 20 | 2.5 | 12,920.369 | 5.16815 | 0.667 h | separate short-length Hχ from Aχ |

The best-amplitude F1 point is 60 µm, α=6. It is not proposed in addition to
the mandatory 58.88 µm, α=6 calculation: the lengths differ by only 1.9%.
Charging two full calculations for those nearly equivalent points would
contradict the purpose of the staged design.

Cost estimates are conservative maxima of:

- a linear extrapolation of measured positive-F2 time against α;
- the median measured F2/F1 time ratio applied to the nearest F1 point.

The total estimated sequential time is 9,584.7 s, or 2.66 h. These are
planning estimates, not solver guarantees.

The manifest contains complete commands but marks every candidate
`proposed_not_run`. A replacement proposal requires a regenerated immutable
F1 collection with the Newton-25 policy declared consistently, followed by
new human approval.

## Temporary scientific conclusions

The present evidence supports the following statements:

1. the DCT frozen-field response is useful for rejecting and ordering large
   portions of parameter space;
2. factor-two F1 mechanics preserves the ranking of the existing P43 F2
   calculations within the pre-declared error gates;
3. stronger coupling continues to improve the amplitude objective over every
   converged profile;
4. localization is not monotone in exactly the same way, producing a genuine
   Pareto trade-off;
5. short-length/high-coupling points require more mechanical Newton
   iterations, but the former \(\alpha=2.5\) boundary was not a physical or
   micromorphic convergence limit;
6. the current data do not provide an interior optimum or demonstrate
   separate identifiability of \(H_\chi\) and \(\ell\);
7. no material-length conclusion is allowed.

After approved F2 calculations, at most three non-dominated pairs may be
frozen and transferred to another band-containing ROI. That transfer must use
the same observation operator and metrics, with no recalibration.

## Reproducibility and cache rules

Every result key contains the relevant mesh, DIC, material, configuration,
loading history, observation operator, fidelity, constitutive variant,
\((\alpha,H_\chi,\ell,A_\chi)\) and Git revision. Results are reused only when
all physical and numerical hashes agree.

The consolidated table distinguishes:

- `F0_frozen`: heuristic screening only;
- `F1_low`: validated ranking only;
- `F2_high`: scientific full-resolution result.

The compact tables, profiles, proposal and transfer protocol used by this page
are versioned under
`validation/reference_data/joint_nonlocal_identification_p0043_v1`.

The complete operational sequence is given in
{doc}`../how-to/run_joint_nonlocal_identification`.
