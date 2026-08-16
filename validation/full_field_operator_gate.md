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

## Registered scope of this milestone

Full field, 3599x3099, **isotropic homogeneous elasticity only**. No CNN, no
assembled-field projection, no training, no EBSD, no local coefficients. One
question:

```text
eps_p  --A-->  d eps      on the real field, with the measured Dirichlet data
```

## Registered tasks

1. Build the lifting of the measured Dirichlet data, `u = u_D + u_tilde` with
   `u_tilde` vanishing on the boundary, so the correction problem is
   `K_II u_tilde_I = f_I(eps_p) - K_IB u_D`.
2. Implement `A` matrix-free over the whole domain.
3. Implement `A^T` **separately**, never assumed equal to `A`.
4. Qualify the pair: `|<Ax, y> - <x, A^T y>| / (|Ax| |y|)` below **1e-8**, on
   the real domain with the real Dirichlet treatment, over several random pairs.
5. Measure time and peak memory of `A` and `A^T`.
6. Test whether the spectral operations fuse into a single symbol. `A = B K^-1
   B^T w C` admits one only while `K` is translation invariant, which holds for
   homogeneous elasticity and will not hold once per-point crystalline stiffness
   enters. The gain is real now and must not be built upon as permanent.
7. Compare at least two Dirichlet strategies: a fast direct spectral route
   (sine/cosine diagonalisation, available after the lifting makes the boundary
   conditions homogeneous) against a matrix-free iterative solve preconditioned
   by the homogeneous spectral operator.

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

**The FFT investment is not disposable when crystallography arrives.** With
`C(x) = C_0 + Delta C(x)` the homogeneous inverse `K_0^-1` remains the natural
spectral preconditioner of the heterogeneous problem, which is the whole basis
of heterogeneous FFT methods. And elastic anisotropy need not enter with the
EBSD at all: isotropic elasticity with crystallographic geometry in the *plastic*
generator separates the two effects cleanly and should be tried first.
