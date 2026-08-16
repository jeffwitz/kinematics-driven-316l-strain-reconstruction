# DIC-driven plastic identification: the whole approach, for a cold restart

This document exists so the work survives a lost session. It states the
problem, the mechanical chain, what has been established, what has been
refuted, where the data lives, and what to do next. Anything asserted here has
a script and an artefact behind it; anything uncertain is marked as such.

---

## 1. The problem

A 316L specimen is loaded monotonically. Full-field DIC gives the displacement
at 41 states. We want the plastic strain field and, eventually, the
constitutive law that generates it -- **from the measurements and equilibrium**,
without assuming a law in advance.

The measured kinematics is the only observable. There is no stress measurement,
no load cell series (searched for; absent), and no ground truth for the plastic
field. Everything therefore passes through the mechanical operator below.

---

## 2. The mechanical chain

### The eigenstrain forward operator

Treat plasticity as an eigenstrain. With `C` the plane-stress elasticity, `B`
the strain operator and `K` the stiffness restricted to interior degrees of
freedom, a plastic field `eps_p` produces, at fixed boundary data,

```text
A eps_p = B K^{-1} B^T w C eps_p
```

that is, the strain response to the eigenstrain forcing. Implemented in
`identification/tensor_plastic_observability.py`, matrix-free, with `K`
factorised after being recovered by colouring the 3x3 nodal stencil.

The elastic reference for a state is the field carrying the **same Dirichlet
data** with no plasticity:

```text
eps_el(n) = B u_el(n),     u_el interior corrected so that K u_el = 0
```

and the defect the inverse must explain is `r_n = eps_DIC(n) - eps_el(n)`.

### The property that governs everything

**`A` is surjective onto zero-boundary fields.** Any measured increment can be
reproduced exactly by some eigenstrain, and LSQR does so in a handful of
iterations. Therefore:

> fitting the DIC proves nothing. Only the constraints imposed on the plastic
> field carry information.

Every apparent success of an unconstrained inverse in this project is explained
by this one fact. It is the single most important thing to remember.

### Conventions, and the traps in them

* **Kelvin/Mandel** internally: `[e_xx, e_yy, sqrt2 e_xy]`, so the Euclidean dot
  product *is* the tensor contraction and `sigma : d eps_p` needs no metric.
  `core/kelvin.py`. An engineering `B` converts by **dividing** the shear row by
  `sqrt(2)`, not multiplying.
* **`PLANE_STRESS_PLASTIC_GAUGE`** has eigenvalues 2/3, 2/3, 2. Kelvin does not
  make it the identity: with `eps_zz = -(eps_xx + eps_yy)` the plane-stress
  plastic triple is not an orthonormal subspace of the deviatoric space. `p_eq`
  is `sqrt(z^T G z)`, never `norm(z)`.
* Plastic incompressibility is **built into the representation**, so it needs no
  constraint: the three in-plane components imply the third.
* In plane stress `sigma_zz = 0`, so the in-plane Kelvin dot product is the
  whole of `sigma : d eps_p`.

### Kinematics

`TwoSubcellDiagnostic2D` -- two triangular subcells per pixel, two independent
constitutive histories. Measurement, simulation and inverse all live natively in
`(nx, ny, 2, 3)`, so **no interpolation is needed anywhere**. The two-state
solver `spectral2d/newton_two_state.py::solve_two_state_dirichlet_plane_stress`
uses the same kinematics; it gained an optional `increment_observer` returning
displacement, stress and plastic strain of every converged increment.

Do not confuse it with `EBITwoTriangleKinematics2D`, which shares **one**
material state between the two triangles.

---

## 3. The data

### Locations

Large datasets live outside the repository, in
`/home/jeff/CNRS/Theses/Adil/essais/9_numerical/`, and are destined for a forge
once curated. Only scripts, notes, small JSON and figures are committed.

| file | content |
|---|---|
| `DIC_images/000294.tif` ... `000335.tif` | 42 raw speckle images, 4400x5400 |
| `p0043_disflow_history_tuned.h5` | **the reference history**: 41 states, 3600x3100, converged settings |
| `p0043_evm_history.h5` | equivalent strain of the above, plus its noise map |
| `p0043_disflow_history.h5`, `..._patch4.h5` | earlier settings, kept to measure the price of the parameters |
| `CP_dataset.h5` | EBSD orientations, max Schmid factor, and the received `U_40`/`V_40` |
| `fem_partition_E200_allframes_dataset.h5` | **Abaqus**, 18 frames -- simulation, not measurement |

`000294.tif` is the undeformed reference, `000295`-`000334` are steps 1-40, and
`000335` repeats the final state. That repeat is the null test.

The crop from raw image to prepared field is `rows[400:4000],
columns[1211:4311]`, verified on a derivative quantity by a shift scan with its
optimum at zero. Traction runs along **image columns**; `U_40 = u_y` correlates
at 0.99996 with the column displacement, `V_40 = u_x` with the row
displacement. Pixel size 1.84 um.

### The DISFlow settings, and why

`finest_scale` 0, patch 4, stride 1, `alpha` 15, `epsilon` 0.01, **100**
refinement iterations.

The iteration budget was the dominant term, not the smoothing weight and not the
correlation window. At thirty iterations the matching stage's patch grid is
still visible as a checkerboard -- which the received `U_40`/`V_40` also carry,
so **those fields are unconverged**. At `epsilon` 1e-3 the penalty is close
enough to total variation to produce staircase bands that a derivative turns
into spurious localisation. Neither parameter was chosen to resemble the
received fields, precisely because those carry an artefact.

### The noise floor, measured

From the repeated final state, at the settings above: **0.148 pixel** on
displacement (0.27 um), and on strain a noise-to-signal ratio of 0.073 in xx,
0.072 in yy, 0.183 in xy, **0.100** on the equivalent strain. Signal exceeds
noise down to a **two-pixel** wavelength, so nothing in the metrology limits the
spatial scale of the analysis.

More refinement iterations lower the strain noise while preserving structure,
so the settings that resolve more are also the cleaner ones.

---

## 4. What is established

* **A plastic eigenstrain of realistic amplitude reproduces each measured
  increment exactly.** Uninformative on its own, by surjectivity.
* **Ludwik/J2 with the registered per-pixel yield map makes the agreement
  worse.** `E_L = |eps_L - eps_DIC| / |eps_el - eps_DIC|` is 1.64, 1.70, 2.07,
  2.91 at states 25, 30, 35, 40. The amplitude is right -- mean equivalent
  strain within 2 % -- but the distribution is wrong: coefficient of variation
  0.77 against 0.22 measured, correlation with the measurement 0.229 where the
  purely elastic solution reaches 0.645. The model localises into the soft
  pixels of the yield map (correlation -0.569) and the specimen does not
  (-0.196). The correction it applies is **orthogonal** to the defect, cosine
  +0.006 to +0.038, so no rescaling can help.
* **The elastic defect is real and well above noise**: 0.29 of the measured
  strain norm at state 40 against a noise ratio of 0.100.
* **The strain follows the Schmid factor weakly but monotonically**, +0.008 at
  state 1 to +0.116 at state 38, positive throughout, carried mostly by the
  loading mode. An artefact would be constant. EBSD registration is declared but
  not verified, which is the dominant uncertainty.
* **The EBSD carries 827 non-indexed pixels**, 0.0074 % of the map, filled with
  the sentinel 1449 in all three Euler angles. They are invisible to any
  orientation-derived quantity -- a sentinel still yields a valid rotation and
  therefore a plausible Schmid factor -- but the archived `max_schmid_factor`
  exposes them, since a Schmid factor must lie in `[0.2722, 0.5]` and theirs do
  not. That lower bound is the `<111>` loading direction, which minimises the
  largest of the twelve factors. `schmid_channels` uses the archived map purely
  as this detector and refills the offending pixels from their nearest valid
  neighbour, carrying a validity flag alongside.
* **The loading axis is the first sample axis, measured not assumed.** `g` maps
  sample to crystal, so the axis the crystal sees is a *column* of `g`. Columns
  0, 1 and 2 reproduce the archived map at 0.77, 0.18 and 0.06. The
  acquisition notes do not record the convention; this comparison is the only
  evidence, and 0.77 with 18 % of pixels agreeing to 1e-3, no shift improving
  it, and near-identical distributions (medians 0.4633 and 0.4615) says the
  residual is a convention inside their calculation rather than a registration
  offset. Ours is bounded correctly by construction and is what is used.
* **The EBSD-to-DIC frame is now verified rather than declared**, which the
  inventory listed as the dominant uncertainty. Three independent measurements,
  none of which relies on the archived map:
  - The DIC tensile axis is the **column** axis, `d(u_col)/d(col)` reaching
    +0.224 % at state 40 while `d(u_row)/d(row)` contracts to -0.187 %.
  - Correlating our Schmid map against the measured field at state 38 ranks the
    three sample axes at +0.119, +0.062 and -0.028, selecting axis 0. Sample X
    therefore lies along the image columns: the sample frame sits a quarter turn
    from the array indexing, the ordinary EBSD convention.
  - The grids need no transformation. Every mirror collapses the correlation
    (+0.119 to +0.030, +0.052, -0.009), and a shift scan over +/-64 px peaks at
    (0, -4) with +0.1227 against +0.1220 at zero -- a plateau, not an offset --
    falling to +0.062 at the edge. Registration holds to a few pixels, well
    below the grain size.
* **The loading is macroscopically tensile but the field is not strictly
  uniaxial**, the in-plane transverse-to-axial ratio of the total strain being
  0.835 at state 40. This is a characteristic of the test, not a defect, and
  there is **no general 0.5 bound on that ratio** -- an earlier revision of this
  document claimed one and was wrong. Plastic incompressibility says
  `tr(eps_p) = 0`; the one-half follows only when the two transverse components
  are equal, and 2D DIC does not measure `eps_zz`, which is free to take
  -0.165 `eps_xx` and close the trace exactly. The measured numbers are total
  strains besides, so comparing their ratio to a property of `eps_p` is a
  category error. Real grips, local multiaxiality and 316L texture all pull the
  in-plane ratio away from any idealised value. Nothing here needs
  investigating.
* **The fine 2-8 px band is not a fixed pattern**: correlation 0.820 between
  neighbouring states decaying to 0.188 between extremes, with the across-state
  mean holding 47.8 % of its energy. Half common, half evolving.

## 5. What is refuted, including my own errors

* **The elastic lifting used across the P43 scripts does not equilibrate**, and
  this is a defect in the code rather than a result. Three conventions meet in
  one expression: `operator.elasticity` is the **Kelvin** stiffness,
  `kinematics.strain` returns **engineering** shear, and
  `divergence_from_sample_stress` expects **Voigt**. The `extension()` helper
  chains them without conversion, which doubles the shear stress. Measured on a
  non-equilibrated field, the lifted result retains **32 %** of the interior
  equilibrium residual where the converted form reaches 4e-16, and the two
  lifted fields differ by 92 % in norm. The operator chain itself is sound --
  `matvec`, `rmatvec` and `kelvin_response` convert correctly -- so what is
  affected is the elastic reference, and therefore every residual measured
  against it: the "elastic defect is 0.29 of the measured strain norm" figure,
  the residual-driven Krylov subspaces, and the dissipation studies built on
  them. Verified numerically for one script; the same textual pattern appears in
  roughly a dozen others under `scripts/*_p43.py`. Not yet repaired, and no
  campaign replayed: recorded here so that no conclusion drawn from those
  residuals is quoted again without redoing the lifting.
  `adaptive_reduced_basis_p43.py` converts and asserts the residual at startup.
* **Morphological inpainting adds nothing over a Laplace solution**, which is
  what stopped the campaign. With the pooling defect repaired and a genuine
  spatial decoder, the network reaches a morphology error of 0.997 to 1.008
  across contexts 32, 64 and 128 while harmonic inpainting sits at 0.956 to
  0.987 -- a gain of -1.3 % to -5.4 %, that is, a loss. Runs stopped mid-sweep
  and kept as a negative control on the fixed-basis premise, not as evidence
  about the adaptive one.
* **No global reduced representation works.** POD per Laplacian band gives
  held-out errors 0.562, 0.507, 0.544, 0.320 at rank 31 with a training error of
  exactly 0.000 in all four bands. A convolutional autoencoder plateaus at a
  gradient error of 0.97; a CROM-style neural field at 1.00.
* **The 0.115 global POD figure was an artefact** of normalising by a field
  close to one everywhere. Comparisons made against it are void.
* **Thirty-two states cannot support a temporal holdout**: rank-31 basis,
  thirty-two snapshots, no statistical room. The power of this dataset is
  spatial.
* **The pointwise dissipation constraint admits only zero** in a
  residual-driven Krylov subspace, at ranks 8, 32 and 128 alike, while the free
  increment fits exactly at rank 8. The subspace is built from `A^T g` and a
  dissipative direction must align with the local stress; those are unrelated
  objects. Not verified: whether the cone is genuinely trivial or the QP
  declines to leave the origin -- a feasibility LP, `max t` subject to
  `G q >= t`, `|q| <= 1`, would decide it.
* Mistakes of mine that cost real time, recorded because they share one shape --
  a conclusion drawn before an order of magnitude was checked: decimating by
  three after measuring signal down to two pixels; a Fourier bandwidth ten times
  too narrow; proposing to low-pass the fine band without testing its temporal
  correlation; quoting 0.115 as a bar; one mistyped slip direction,
  `(1,-1,1)[-1,0,-1]`, whose dot product of -2 pushed the Schmid factors to
  0.905 against a hard bound of 0.5. Each was caught by a bound, never by
  inspection, which is why `FCC_SYSTEMS` now asserts its own orthogonality and
  `schmid_channels` asserts its own range.
* And the converse mistake, which is worse because a bound is what stops the
  others: **inventing one**. The transverse-to-axial ratio was reported as
  violating a physical maximum of 0.5 that does not exist, on strains that were
  not even the plastic ones. A bound is only usable when its hypotheses are
  written down next to it -- 0.2722 for the largest Schmid factor holds for any
  FCC orientation whatsoever; 0.5 for the strain ratio holds only under
  idealised uniaxial tension with a symmetric transverse response, which this
  test does not have.

---

## 6. The current hypothesis

Not "the field admits a low-dimensional global representation" -- that is
refuted -- but:

> **is there a spatially transferable local rule which, coupled to equilibrium
> and thermodynamics, reproduces the measured heterogeneity?**

```text
d eps_p(x, t) = F_theta( mechanical state around x, history around x )
```

with the **same** `F_theta` everywhere. The global field may then have an
enormous dimension; what must be compact is the rule that generates it, applied
to local states that equilibrium supplies. Long-range interaction is the
solver's job, not the network's -- which is the argument against a transformer
or a global implicit representation.

### The object being reduced was wrong, and the correction

The failures listed in section 5 share one premise, and it is the premise rather
than the methods that is now identified as the fault. **Nothing requires the
plastic field to lie on a global low-dimensional manifold.** What must be low
dimensional is the space of admissible plastic corrections *around the current
mechanical state*:

```text
refuted     eps_p_n = Phi a_n             one fixed Phi(x) for every state
registered  eps_p_n = Phi_theta(S_n) a_n  a basis the current state generates
```

The inverse still solves for `r` numbers per increment, but the basis they
weight is regenerated at each one and is free to move a band, reorient it, or
follow the microstructure. The `r` numbers no longer encode an image; they
weight directions the mechanics itself produced. Ranks 2 to 32, on the P43
100x100 demonstrator first, where `TensorPlasticObservabilityOperator` already
supplies `matvec`, `rmatvec` and `kelvin_response` in Kelvin.

Two failure modes specific to this formulation, both registered in
`adaptive_reduced_basis_preregistration.md`:

* **The generator can hide the answer in the basis.** With enough capacity a
  rank-one `Phi` equal to the required correction gives `E_DIC = 0` at `r = 1` --
  the surjectivity of `A` returning one level up. `S_n` is the predictor state
  and excludes the interior DIC, but the coefficients must also be fitted on the
  training region alone, since `a_n` is global and a full-field `g_n` would
  otherwise pull the held-out region into the fit.
* **The reduced dissipation cone is expected to be trivial.** `C a >= 0` is
  millions of half-spaces intersected in `R^8`; the Krylov cone already admitted
  only `q = 0` at every rank. The remedy is structural rather than constrained:
  modes built as `m_k(x) N(sigma_bar)` with `m_k >= 0` are dissipative by
  construction, and the crystallographic form
  `sum_alpha gamma^alpha sign(tau^alpha) P^alpha` is dissipative *and* makes the
  EBSD ablation structural instead of an extra input channel.

The loop, at state `n`:

```text
eps_p(n-1) and the DIC Dirichlet data
  -> equilibrium              -> sigma, eps everywhere
  -> local patches            -> F_theta -> d eps_p in Kelvin
  -> incompressibility, dissipation
  -> equilibrium again        -> eps_sim
```

with `eps_sim` compared to `eps_DIC` **in the loss only**, and never fed to
`F_theta` inside the evaluation region.

### Why this is not circular, and where the guarantee comes from

Equilibrium does **not** regularise: `A` is surjective, so a fit is guaranteed.
What breaks the degeneracy is **weight sharing** -- one function used at ten
million places -- together with incompressibility and positive dissipation.
That is a statistical guarantee, not a mechanical one, and must be stated as
such.

### Guardrails

* **Local inpainting is not a local law.** A network shown a large context and
  asked for a small core can interpolate. Any morphological transfer test needs
  a bicubic or kriging baseline; beating it proves a transferable local
  regularity and nothing more.
* **P43 and the small crops qualify the software, not the hypothesis.**
  Spatial transferability can only be established on the full field.
* **Do not give absolute `(x, y)` to the network.** It would learn a map of the
  specimen rather than a law. This is the most obvious leak.
* Do not materialise a patch dataset: sample states, positions and windows
  online from the HDF5.
* **The measured boundary conditions are the problem statement.** Never rescale,
  symmetrise or otherwise correct the DIC displacements to bring them closer to
  an idealised loading -- the transverse contraction, any grip rotation, the
  local multiaxiality and the anisotropy are what makes this dataset worth
  solving against. Using the real conditions rather than an invented uniaxial
  tension is the point of the approach, not an imperfection in it.
* **Do not assert a bound without its hypotheses.** The two in play here are not
  alike: 0.2722 to 0.5 for the largest Schmid factor holds for every FCC
  orientation and caught two real defects; 0.5 for the transverse-to-axial
  strain ratio holds only under idealised uniaxial tension and was pure
  invention here. A bound whose assumptions are not written beside it will
  eventually manufacture a false anomaly.

### Milestones, in order

Milestones 1 and 2 -- morphological inpainting and the useful-radius ablation --
were **stopped mid-run and archived as negative controls**, together with the
POD, CAE and INR campaigns. They tested how well a network reconstructs an image
from its surroundings, which is no longer the question. The partial results,
network morphology error 0.998 to 1.008 against harmonic inpainting at 0.876 to
0.986, are recorded in section 5 and settle nothing about the reduction
hypothesis in its current form.

1. **Reduced adaptive basis on the P43 100x100 demonstrator**, per
   `adaptive_reduced_basis_preregistration.md`: `E_DIC(r)` held out, against the
   free plastic field above and the fixed POD/Krylov basis below, at ranks 2 to
   32, with the free and the stress-aligned mode parameterisations.
2. **Structural EBSD ablation** -- crystallographic modes against isotropic
   normal modes, same architecture, same rank, same optimiser.
3. **Adjoint qualification of the full-field solver** -- see below. Still ahead
   of any full-field work, and now also needed to differentiate the reduced
   least squares through `A Phi`.
4. **Full field**, conditional on milestone 1 succeeding: `A Phi` costs `r`
   operator applications per increment, which is the reason `r` must stay small.
5. **Spatial then temporal validation**, with no interior DIC given to
   `F_theta` and the coefficients fitted outside the held-out regions.

### The item whose cost is unknown

Milestone 3 gates 5 and 6 and is the only one that cannot be estimated. Each
training step needs a global equilibrium solve and its adjoint. A sparse
factorisation is viable at twenty thousand degrees of freedom, not at
twenty-two million, so the FFT Green operator in `spectral2d` is the tool. But a
periodic FFT Green function does not by itself solve non-homogeneous Dirichlet
data; a lifting `u = u_D + u_tilde` with `u_tilde` vanishing on the boundary is
needed, and nothing guarantees a priori that the adjoint of the resulting primal
is consistent. Qualify it with a dot-product test,
`|<Av, w> - <v, A^T w>|` relative, below 1e-8, before any full-field training.

---

## 7. Where things are

Notes: `validation/morphology_reduction_findings.md` (the reduction campaign),
`validation/ludwik_on_the_measured_p43_history.md` (the Ludwik verdict), and the
older `validation/tensor_plastic_observability_m20.md`,
`validation/dic_excitation_of_observable_plastic_modes.md`,
`validation/dirichlet_crop_and_the_noise_reference.md`.

Artefacts: `validation/_generated/cnn_morphology_benchmark/` and
`validation/_generated/disflow_profiles_p43/`.

Scripts, roughly in the order they matter: `compute_full_field_dic_history.py`,
`measure_null_test_noise_floor.py`, `build_evm_history.py`,
`benchmark_pod_per_scale.py`, `replay_ludwik_two_state_history_p43.py`,
`build_incremental_dissipative_history.py`, `correlate_modes_with_schmid.py`,
`test_fine_band_is_fixed.py`, `morphology_benchmark_split.py`.

Abandoned but kept: `benchmark_cnn_morphology.py`,
`benchmark_neural_field_morphology.py`, `build_lowpass_evm_history.py` (written,
never used -- the fine band must not be filtered).

## 8. Standing constraints

French for discussion, English for code, comments, documentation and commits.
Never move a preregistered threshold after seeing results, and keep negative
results. Do not replay expensive campaigns or invent missing coefficients.
Every unit of work ends with a `Claude.md` update and an explanatory commit.
Large results never enter the repository. A concurrent agent session edits this
repository in parallel: re-survey before committing.
