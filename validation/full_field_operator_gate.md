# Milestone 0 — the full-field mechanical operator

Registered before any implementation. This is a **correctness gate, not a
speed benchmark**, and nothing downstream is built until it passes.

## Why this comes first, and why P43 stops being the object

P43 at 100x100 was a demonstrator for a hypothesis and it served. It is not a
prototype for scale, and continuing to refine it was becoming an increasingly
good answer to a question nobody will ask of an 11-megapixel specimen. It
reverts to what it should have stayed: a unit test.

Two arithmetics forced the change.

**The mechanics does not scale as implemented.** The demonstrator uses a sparse
direct factorisation, 12 ms per solve at 19 602 interior degrees of freedom. In
2D that costs about `N^1.5`, so 22.3 million degrees of freedom is a factor of
38 000 -- roughly 460 s per solve, with a factorisation that does not fit in
memory. An FFT Green operator is the alternative: a 3600x3100 real round-trip
measures **616 ms**, so an application of `A` costs a few seconds.

**The reduced coefficients do not scale physically.** Sixteen global numbers
describe a plastic increment over a few grains plausibly and over thousands of
grains not at all. The computational wall and the modelling flaw are the same
wall, which is the useful part of the finding.

## What was wrong in the first cost estimate

Building `A Phi` column by column costs `r` applications of `A`, giving
`16 x 20 x 2 x 3 s` = 32 minutes per gradient step and 4.4 years for a campaign.
That figure is an artefact of forming the reduced normal equations explicitly,
not a property of the problem. Assemble the field first,

```text
v = sum_jk a_jk w_j phi_jk  ->  q = P_H(v)  ->  r = A q - g
p = A^T W r                 ->  dJ/da_jk = <DP_H(w_j phi_jk), p>
```

and the mechanical cost of the whole gradient is **one `A` and one `A^T`**,
whatever the number of coefficients. Thousands of local coefficients cost
thousands of local contractions, not thousands of global solves.

Two honest qualifications on that.

The *inner* problem still has to be solved. With sixteen coefficients the normal
equations are formed in closed form; with fifty thousand they are not, so `a` is
found matrix-free and every iteration costs `A + A^T`. Warm starting from the
previous increment and from the previous optimisation step, plus a semi-smooth
active-set exploiting that `P_H` is piecewise linear, are what keep that
iteration count small; assuming 20 to 50 restarts from scratch would be
pessimistic.

The *outer* gradient probably does **not** need a second adjoint. If the outer
loss is exactly the inner objective, the envelope theorem removes
`da*/dtheta` entirely. Checked against the current code: the DIC term of the
outer loss *is* the inner data term, the orthogonality penalty does not depend
on `a` at all, and only two things break the identity -- the ridge, which
contributes `O(ridge)` with `ridge = 1e-6`, and the dissipation penalty at
weight `1e-2`. That penalty is **redundant in the constrained arms**, where
dissipation holds by construction. Removing it makes the envelope theorem apply
to `O(ridge)` and halves the budget. To be confirmed numerically when the outer
loop is built.

## The full-Dirichlet solver already exists. Do not rebuild it.

An earlier revision of this document registered "build the lifting", "compare
two Dirichlet strategies" and "test whether the spectral operations fuse". All
three were written without reading `docs/explanation/spectral_mechanics/`, and
all three are already answered there and in the code.

```text
u = u* + u^f,     u*|_boundary = u_DIC,     u^f|_boundary = 0
```

`u*` is a discrete harmonic extension of the measured boundary displacements --
`spectral2d/boundary.py::HarmonicDirichletExtension2D` -- so the transform acts
on the homogeneous fluctuation `u^f` and the problem is **not** periodic. The
production chain is local kinematics, then `sigma` and `C_alg`, then
`R = -sum B^T sigma`, solved by a matrix-free Newton-GMRES with
`J v = -sum B^T C_alg B v`, and the DST-I basis -- compatible with `u^f = 0` on
all four edges -- applies `B_0^-1` as the **preconditioner**. No global plastic
stiffness is ever assembled. `full_dirichlet_formulation.md` is explicit that
`B_0` is a Gelebart-type reference operator and deliberately *not* the exact
inverse of the coupled isotropic stiffness.

So the scaling problem was never "find a fast Dirichlet solver". It is:

```text
plug the new plastic inverse onto the spectral Dirichlet solver that exists
```

## Registered scope of this milestone

Full field, 3599x3099, **isotropic homogeneous elasticity only**. No CNN, no
assembled-field projection, no training, no EBSD, no local coefficients, and
**no new boundary treatment**. One question:

```text
eps_p  --A-->  d eps      on the real field, reusing the existing solver
```

## Registered tasks

1. Implement the matrix-free action `A: d eps_p -> d eps` on top of the existing
   machinery: the plastic field enters as a mechanical source, the fluctuation
   `u^f` is obtained by the existing preconditioned Krylov solve, and the strain
   is read back. Reuse the harmonic lifting, the homogeneous fluctuation
   unknowns and the DST-I `B_0^-1` preconditioner as they stand.
   Note this solve is **linear** -- our `A` is the elastic eigenstrain response,
   not the constitutive Newton loop -- so it is one preconditioned Krylov solve
   and no Newton iteration. `K = B^T C B` with Dirichlet data is symmetric
   positive definite, so the solver is **preconditioned conjugate gradient**, not
   GMRES: GMRES belongs to the constitutive Newton loop where the Jacobian is
   neither symmetric nor definite. CG is cheaper per iteration, stores no Krylov
   basis, and its symmetry makes the adjoint cleaner. Both are equally
   matrix-free -- each needs only `v -> K v` and `r -> M^-1 r` -- but CG has a
   short recurrence, so three working vectors against the fifty full-field
   vectors a GMRES(50) would hold at 22 million degrees of freedom.

   What CG requires in exchange is symmetry and positive definiteness of **both**
   the operator and the preconditioner, and that is verified, not assumed.
   `K = B^T C B` restricted to the interior is SPD in linear elasticity, a
   principal submatrix of an SPD matrix being SPD. `B_0^-1` is diagonal in modal
   space and SPD if its symbol is positive throughout -- `green.py` already
   validates `mu > 0` and `lambda + mu > 0` on the reference Lame parameters,
   which is a good sign that positivity was considered -- but the discrete
   implementation can still break symmetry, typically if the interior grid size
   does not match what the DST-I assumes. Test `<K v, w> = <v, K w>` and
   `<v, M^-1 v> > 0` on random vectors before trusting CG; the fallback is GMRES
   or an explicit symmetrisation. A three-line test is a better way to find this
   out than a CG that quietly diverges on the full field.

   The inversion is **not** done in Fourier and cannot be. DST-I does not
   diagonalise the coupled elastic operator under zero Dirichlet on all four
   edges: `(lambda + mu) grad(div u)` carries mixed derivatives, and with
   `u_x` expanding in `sin.sin`, `d2 u_y / dx dy` produces `cos.cos`, which
   leaves the space -- no single separable transform can do it, since a cosine
   basis cannot vanish at both ends. In the *periodic* case `Gamma_hat(k)` would
   be exact, a matrix multiply per wavevector handling the coupling without
   approximation; the boundary is what breaks that, not the elasticity. Hence
   `B_0` as a reference operator, hence a preconditioner, hence iterations.
2. Implement `A^T` **separately**, never assumed equal to `A`. `K` is symmetric
   so `A^T = C B K^-1 B^T w` reuses the same solve, but the implementation is
   written and tested independently.
3. Qualify the pair: `|<Ax, y> - <x, A^T y>| / (|Ax| |y|)` below **1e-8**, on
   the real domain with the real Dirichlet treatment, over several random pairs.
   **The Krylov tolerance and this threshold are coupled**, and not merely as a
   matter of accuracy: an iterative solve to tolerance `tau` is not a linear
   operator at all, since the Krylov subspace depends on the right-hand side. So
   the identity cannot be verified better than `tau`, and the solve must be
   tightened well below 1e-8 or the test measures the stopping criterion rather
   than the adjoint. The transform inside the preconditioner is applied
   identically in both directions.
4. Measure `T_A`, `T_{A^T}` and peak memory, and report the **Krylov iteration
   count**, which is what actually sets the cost: `B_0` is a reference operator,
   not the exact inverse, so even the linear elastic solve iterates. That count
   is the number to know before any budget is written down.

## Measured, at the gate

Mesh independence, which is the result that decides feasibility: **21 PCG
iterations** at 24, 48, 100 and 200 pixels square, invariably, against 178, 352,
717 and 1398 unpreconditioned. The problem does not get harder with size; the
kernel is simply expensive.

The sign convention was caught by the positivity check, not by reading: both
operators are symmetric to 2e-15, but `B_0^-1` follows the production
convention where it acts on `R = -sum B^T sigma` while the stiffness here
already carries that minus, so `M` came out uniformly negative definite and the
composed system indefinite. Fixed, and the gate then passes: adjoint 2.3e-15
over four pairs, agreement with the sparse direct operator 4.2e-13.

**Where the time goes, which is not where it was assumed.** At 400 square, with
FFTW and four workers:

| | per call | share |
|---|---|---|
| `K = B^T C B` | 76.9 ms | 84 % |
| `M = B_0^-1` | 14.2 ms | 16 % |

and inside `K`: divergence 54.4 ms, strain 23.5 ms, stress 1.3 ms, unpack
0.9 ms. The transform is a sixth of the cost. FFTW single-threaded matches SciPy
exactly at 200 square (0.60 s per apply either way) and four workers buy 23 %;
running the first sweep on the SciPy prototype was an oversight but not the
reason for the cost.

Extrapolated to the full field -- 160 000 pixels at 400 square against 11.15
million -- `21 (K + M) = 1.91 s` becomes roughly **134 s per apply**, some 112 s
of it in `B` and `B^T`.

So the lever is the stencil, not the FFT. On a regular grid with homogeneous
elasticity `B^T C B` is a fixed linear stencil and should be array shifts with
no element arrays and no reassembly. A factor of five to ten there would put
`T_A` under thirty seconds and make the transform the dominant term, at which
point threaded FFTW plans become worth their while.

## Registered acceptance criteria

* The adjoint identity holds below 1e-8 relative. **This is the gate.** A
  periodic FFT Green function does not by itself solve non-homogeneous Dirichlet
  data, and nothing guarantees a priori that the adjoint of the lifted primal is
  consistent. If it fails, nothing downstream is built until it does.
* `A` and `A^T` each complete on the full field within the machine's memory, and
  their cost is reported. No threshold is registered on the time: this is a
  correctness gate and a measurement, not a race.
* A known solution is reproduced. The 100x100 crop, solved by the sparse direct
  operator already qualified at 1.5e-15, must agree with the full-field operator
  restricted to that crop under the same boundary data, to solver tolerance.

## What follows, and only after

* **Milestone 1** -- the assembled-field projection at full field with the
  current generator **frozen**, evaluated by tiles with a halo of the receptive
  radius, and a small number of local coefficients. Even with the sixteen global
  coefficients known to be inadequate, this establishes that the chain scales.
* **Milestone 2** -- local coefficients under a partition of unity with a
  **single global mechanics**, sweeping the coefficient density over hundreds,
  thousands and tens of thousands, to trace `E_DIC` against `N_a` and locate the
  diminishing return. Not to find the best number.
* **Milestone 3** -- matrix-free training with temporal mini-batches, warm
  started `a`, tiled CNN forward and backward, global FFT mechanics. Then the
  real test: `theta` frozen, temporal extrapolation.

## Two things not to repeat

**Windows with DIC imposed on their contour are training material, never
proof.** The boundary kinematics of a window already contains the effect of
everything outside it, so plasticity just beyond the edge can be reattributed
inside. Ten thousand independently solved windows also guarantee nothing about
`B^T sigma = 0` over the whole domain. The architecture is tiled CNN plus a
globally assembled plastic field plus global equilibrium.

**Crystal plasticity does not invalidate CG, but three things must not be
confused.** Symmetry survives heterogeneity: cubic `C(x)` is pointwise SPD for
any admissible crystal, so `K = B^T C(x) B` stays SPD and CG stays valid. What
degrades is the preconditioner, since translation invariance is lost and
`B_0^-1` built on a homogeneous reference no longer approximates `K^-1` as
well -- `n_CG` grows with the elastic contrast, which is a measurable cost
rather than an obstacle, and the repository already carries the cubic constants
(`crystal_plane_stress_elasticity.py`, `srix_parameters.py`) so it need not be
invented. What *would* break CG is putting the algorithmic tangent of crystal
plasticity into the operator: that one is genuinely non-symmetric, through the
hardening interaction matrix and the rotation terms, which is why the production
SRIX and Meric pipeline uses GMRES. Our `A` is not that operator -- the
eigenstrain formulation keeps it linear elastic and pushes all plasticity into
the source. And with isotropic elasticity plus crystallography in the plastic
generator only, nothing changes at all: not the symmetry, not the invariance,
not the preconditioner.

**Do not reason about scale without reading `spectral_mechanics` first.** Two
consecutive plans were drafted around rebuilding a Dirichlet solver that has
been in the repository, documented, for a long time. The bridge page
`docs/explanation/spectral_mechanics/plastic_inverse_reuse.md` exists so the
next reader does not repeat it.

**The FFT investment is not disposable when crystallography arrives.** With
`C(x) = C_0 + Delta C(x)` the homogeneous inverse `K_0^-1` remains the natural
spectral preconditioner of the heterogeneous problem, which is the whole basis
of heterogeneous FFT methods. And elastic anisotropy need not enter with the
EBSD at all: isotropic elasticity with crystallographic geometry in the *plastic*
generator separates the two effects cleanly and should be tried first.
