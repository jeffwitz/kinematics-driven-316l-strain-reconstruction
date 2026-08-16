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
* **The fine 2-8 px band is not a fixed pattern**: correlation 0.820 between
  neighbouring states decaying to 0.188 between extremes, with the across-state
  mean holding 47.8 % of its energy. Half common, half evolving.

## 5. What is refuted, including my own errors

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

### Milestones, in order

1. **Morphological local transferability** -- context ring to hidden core, strict
   spatial holdout, against an inpainting baseline.
2. **Useful radius**, by ablation over context sizes at a fixed core; the plateau
   estimates the informative range of the local rule and sets the architecture.
3. **Adjoint qualification of the full-field solver** -- see below. Moved ahead
   of the P43 loop deliberately.
4. **Mechanical loop on P43** -- primal, adjoint, Kelvin, dissipation, patch
   seams, timings.
5. **End-to-end full-field learning**, DIC in the loss only.
6. **Spatial then temporal validation**, with no interior DIC given to
   `F_theta`.

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
