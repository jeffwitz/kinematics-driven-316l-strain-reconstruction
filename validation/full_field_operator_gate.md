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

## The stencil, identified rather than derived

`K = B^T C B` on this grid with homogeneous elasticity is a fixed local
operator, and it was read off the generic operator's impulse response rather
than rederived by hand -- the generic one is already qualified against the
assembled sparse operator at 4e-13, so it serves as the oracle.

Nothing was assumed. All four pixel-parity classes were extracted and compared,
and the support was allowed to declare its own size.

* **One stencil suffices.** Disagreement between the four parity classes is
  exactly `0.000e+00`.
* **Seven blocks within a radius of one**, not nine: the TRI2 split drops two
  diagonal corners.
* **Reproduces the operator to 1.95e-16** relative, at 64, 128, 256, 512 and
  1024 pixels square alike -- machine precision, not an approximation.
* **Symmetry below 5e-15 and positive energy**, which CG needs as much as it
  needs agreement.

Throughput of the slice-based NumPy prototype, which is a lower bound rather
than the verdict -- each accumulated neighbour rereads and rewrites the whole
output, so seven blocks make seven passes through memory where a fused kernel
makes one:

| pixels | generic | stencil | speedup | Mpix/s |
|---|---|---|---|---|
| 256 | 18.5 ms | 3.3 ms | 5.7 | 19.9 |
| 512 | 91.8 ms | 24.1 ms | 3.8 | 10.8 |
| 1024 | 422.4 ms | 83.7 ms | 5.1 | 12.5 |

Extrapolated to 11.15 million pixels: generic `K` about 4.5 s, NumPy stencil
about 0.9 s, and a fused single-pass kernel plausibly 0.2 to 0.3 s.

**Amdahl then decides when to stop.** With `M` around 1.0 s at full field, a PCG
iteration goes from roughly 5.5 s to 1.9 s on the NumPy stencil alone -- `T_A`
from about 134 s to 40 s. A fused kernel would take it to about 26 s, at which
point the preconditioner is 80 % of the cost and further stencil work is wasted.
That is the moment to switch to threaded FFTW and to reducing the 21 iterations,
in that order, and not before.

## The stencil wired in, and where the time now sits

The generic operator stays as the oracle -- it is the general case, it will
handle heterogeneous `C`, and every change to the fast path is checked against
it. The stencil is a specialised backend valid only under its preconditions,
regular TRI2 grid and homogeneous elasticity, and is *derived from* the generic
one rather than written independently.

The full gate was rerun on both. They agree to the digit: same stiffness
symmetry 3.814e-15, same `v.Kv` 2.847300e+16, same 21 PCG iterations, same
agreement with the sparse direct operator 4.629e-13, adjoint 3.84e-16 against
3.88e-16. Nothing in the chain changed.

| 2048 square | `K` | `M` | `A` | speedup |
|---|---|---|---|---|
| generic | 1749 ms (82 %) | 387 ms (18 %) | 49.7 s | -- |
| stencil | 335 ms (55 %) | 272 ms (45 %) | 17.8 s | 2.80 |

Twenty-one iterations either way, at 1024 and 2048 square alike, and the
stencil holds 12.5 Mpix/s at both sizes.

**The decision this was meant to inform has changed shape.** `K` is at 55 %,
just over the line where a fused native kernel would be worth its dependency,
but only just. A fivefold kernel would take the iteration from 607 ms to 339 ms,
so `A` from about 47 s to about 26 s at full field -- a factor of 1.8.

Cutting the iteration count is now worth as much and costs no dependency at all.
Twenty-one down to ten is a factor of 2.1, and unlike the kernel it divides `K`
and `M` **together**. The coupled `2x2` spectral preconditioner, which the
current `B_0` deliberately is not, is therefore the better next move: same
benefit, no new build chain, and it compounds with a kernel later rather than
competing with it.

Extrapolation is deliberately not carried to 3599x3099 here; the numbers above
are measured.

## `einsum` was tried before any native kernel, and rejected

A sliding-window view contracted by `einsum` should in principle fuse the seven
slice passes into one, without a compiled dependency. Measured at 1024 square,
against the slice version at 97.7 ms and 33.5 MB peak:

| | time | Mpix/s | peak | error |
|---|---|---|---|---|
| `einsum` `(H,W,2)`, `optimize=False` | 3063 ms | 0.3 | 16.7 MB | 9e-17 |
| `einsum` `(H,W,2)`, `optimize=True` | 75.6 ms | 13.8 | 167.5 MB | 1.4e-16 |
| `einsum` `(2,H,W)`, `optimize=False` | 3135 ms | 0.3 | 16.7 MB | 9e-17 |
| `einsum` `(2,H,W)`, `optimize=True` | 58.5 ms | 17.9 | 167.5 MB | 1.4e-16 |

`optimize=False` is thirty-one times *slower*: the generic engine walks the
strided view element by element and fuses nothing. Layout matters as expected,
`(2,H,W)` beating `(H,W,2)` by 30 %. But `optimize=True` buys its speed by
materialising the window -- peak memory 167.5 MB against a field of 17.5 MB,
which is precisely the 3x3 neighbourhood copied out. At full field that is
roughly 1.8 GB of transient allocation per `K`, twenty-one times per `A`,
inside a CG loop.

The gain is 1.67 on `K`, which at 55 % of an iteration is 1.29 overall. Paying
a fivefold memory factor for 29 % is a bad trade, and the slice version stays.
`einsum` is recorded as tried and rejected on measurement, so it need not be
revisited.

## The coupled preconditioner: tried, isolated, and rejected

`B_0` is diagonal per component, so the cross term of isotropic elasticity --
`(lambda + mu) grad(div u)`, which couples the components through `k_x k_y` --
is absent from the reference. Adding it looked like the obvious way to take 21
iterations down to ten, and the iteration count is now the whole cost.

The first attempt read 32 iterations against 21 and the result was **not
usable**: it replaced the repository's diagonal `2 mu_0 L + lambda_0 X` by the
continuous symbol `mu_0 L + (lambda_0 + mu_0) X` *and* added a cross term, so
two things changed at once and the outcome was attributed to one of them.

Redone with the diagonal left exactly as the repository has it and only
`c = alpha sqrt(X Y)` varying, every mode verified positive definite:

| `alpha` | iterations |
|---|---|
| 0, the repository diagonal | **21** |
| 0.25 `lambda_0` | 21 |
| 0.5 `lambda_0` | 21 |
| `lambda_0` | 22 |
| 2 `lambda_0` | 23 |
| `lambda_0 + mu_0` | 29 |
| -`lambda_0` | 22 |

Monotone, best at zero. The 32 decomposes: 29 from the `lambda_0 + mu_0`
coefficient, the rest from the diagonal changed alongside it.

The reason is the one the geometry already gave. The true discrete coupling is
not diagonal in the DST-I basis -- a mixed derivative maps `sin.sin` to
`cos.cos` and leaves the space -- so a cross term that *is* diagonal per mode
models a coupling that does not exist in this representation. `B_0` is not a lazy
approximation of the `2x2` symbol; it is the best approximation **representable**
in the basis the Dirichlet conditions impose.

The reference Lame ratio is not a lever either: 21 iterations across
`lambda_0 / mu_0` from 0.0625 to 8, degrading only at 16. The count is a robust
property of the operator-preconditioner pair.

What remains for the preconditioner is therefore not its symbol but the cost of
applying it -- warmed FFTW plans, threading, the buffered transform path -- and
warm starting across the many solves a campaign performs, which attacks the
count without touching the reference at all.

## What crystalline elasticity would cost, from the registered constants

`c11 = 218300`, `c12 = 144800`, `c44 = 125400` MPa (`srix_parameters.py`):

| | |
|---|---|
| Zener anisotropy `2 c44 / (c11 - c12)` | 3.412 |
| shear modulus, `{100}<110>` to `{100}<010>` | 36.75 to 125.4 GPa |
| Young's modulus, `<100>` to `<111>` | 102.8 to 301.7 GPa |
| **Voigt / Reuss bracket on `mu`** | 89.9 / 63.8 GPa, **1.409** |

The last line is the one that matters. With a homogeneous reference chosen
between the Voigt and Reuss bounds, the contrast a heterogeneous solve faces is
**1.41**, not the 3.4 separating extreme orientations. Heterogeneous FFT schemes
routinely handle contrasts of ten to a thousand. Crystallography will raise the
iteration count moderately and will not break this preconditioner.

## Warm start, tolerance, and warmed FFTW plans

Consecutive solves in an identification loop have nearby right-hand sides, so
the previous solution is a better opening guess than zero. It is off during
qualification -- it makes the accuracy `A` achieves depend on history, which the
adjoint test must not see -- and on in production.

On a sequence where the plastic field moves by a halving step, eight solves:

| `rtol` | cold | warm | gain |
|---|---|---|---|
| 1e-12 | 168 | 149 | 1.13 |
| 1e-9 | 128 | 109 | 1.17 |
| 1e-6 | 88 | 69 | 1.28 |
| 1e-4 | 56 | 40 | 1.40 |

Warm starting alone is modest, and for a clear reason: it saves the first few
orders of magnitude of residual reduction, so its value is set by the ratio of
the tolerance to the initial residual. **Loosening the tolerance is the larger
lever**, and the two compound -- 168 iterations at 1e-12 cold against 69 at 1e-6
warm, a factor of 2.4, and 21 per solve down to 7. That is the factor hoped for
from a coupled preconditioner, obtained with no new machinery.

The tolerance is a training-time choice and not free. Qualification stays at
1e-12, and so does any solve whose result is reported: the DIC noise floor is
10 %, so a gradient does not need twelve digits, but a measurement does.

Applying the preconditioner, with planning taken out of the timer, at 1024
square:

| backend | workers | `M` |
|---|---|---|
| scipy | 1 | 274.9 ms |
| fftw | 1 | 169.1 ms |
| fftw | 8 | **76.3 ms** |

A factor of 3.6 over the SciPy prototype on what is now the dominant term. The
four-worker point read 577 ms and is discarded: it built in 1.2 s against 16 and
41 for its neighbours, so it loaded stale wisdom instead of measuring a plan.

**Compounded, at 1024 square.** An iteration was `K` 428.6 ms plus `M` 274.9 ms.
It is now roughly `K` 1.4 ms plus `M` 76.3 ms, a factor of **9**, and the
iteration count falls from 21 to 7 under a training tolerance with warm start,
for **27 overall** on `A`. The 134 s extrapolated at the start of this milestone
lands near five.

## FFTW at full field: the size analysis I got wrong, and what it is worth

I blamed the transform cost on `3098 = 2 x 1549` and proposed cropping the
domain. **That factorised the wrong number.** For `RODFT00` the logical length
is `N = 2(n + 1)`, so it is `n + 1` that must factor well -- and with an
interior of `pixels - 1`, that is the pixel count itself.

| interior | `N = 2(n+1)` | scipy DST | ns/point |
|---|---|---|---|
| 3598 x 3098 (ours) | `2.59.61` x `2.3.1033` | 1672 ms | 150.0 |
| 3599 x 3099 | `2^5 3^2 5^2` x `2^3 5^2 31` | **759 ms** | **68.0** |
| 3597 x 3099 | `2^2 7.257` x `2^3 5^2 31` | 1146 ms | 102.8 |

A factor of **2.2**, not the 1.29 an earlier comparison suggested -- that one
was against 3584 x 3072, which is smooth in `n` and poor in `n + 1`.

The smooth interior needs 3601 x 3101 displacement nodes and the data has 3600
x 3100. So the domain is not cropped; the **preconditioner** is padded, which is
legitimate exactly where cropping would not be: `M_pad = R M' R^T` is a
principal submatrix of an SPD operator and therefore SPD, and a reference
operator never had to be exact.

Planning is a one-off and it does terminate. Measured at full field: the plan
took several minutes, `M` then costs **2560 ms** against 6394 for SciPy and 8815
for FFTW with `estimate` -- so FFTW is worth it *only* with a measured plan, and
with `estimate` on this size it loses to pocketfft. Wisdom rebuilds the operator
in 11.3 s rather than replanning.

The 2.2 above is a **SciPy** ratio and transferring it to FFTW was unverified.
Measured properly it is worse than that, in the direction that favours padding.
A controlled pair at a million points, measured plans throughout:

| interior | `N` | scipy | fftw estimate | fftw measured |
|---|---|---|---|---|
| 1023 x 1023 | `2^11` | 41 ms | 7 ms | **8 ms** |
| 1032 x 1032 | `2 . 1033` | 93 ms | 1228 ms | **97 ms** |

FFTW only shines on smooth sizes: it beats SciPy fivefold on the good one and
merely matches it on the bad one, while `estimate` collapses to 1228 ms. And at
the two real full-field sizes, measured plans:

| size | `N` | plan | DST |
|---|---|---|---|
| 3598 x 3098 | `2.59.61` x `2.3.1033` | 33 s | **373 ms** |
| 3599 x 3099 | `2^5 3^2 5^2` x `2^3 5^2 31` | 67 s | **100 ms** |

So one node of padding is worth **3.7** on the transform, not 2.2.

Padding is not free, and the cost is in iterations:

| pad | iterations |
|---|---|
| 0 | 21 |
| 1 | 29 |
| 2 | 36 |

Reproduced at 200 and 256 pixels square, positive definite throughout. One node
of padding buys 3.7 on the transform and costs 1.38 in count, so **2.7 net**:
`T_A` from about 54 s to about **20 s**. Two nodes would have to buy another 1.24
to break even and will not.

One further thing the timing exposes. A single plan measures in 33 to 67
seconds, yet building the repository's operator took minutes -- so it creates
several plans. `RODFT00` is its own inverse up to scaling, and one plan serves
both displacement components through the new-array execute interface, so there
is probably a factor of two to four to recover on planning alone.

An earlier claim that this milestone would land near five seconds is withdrawn.
It extrapolated a 1024-square measurement linearly and ignored that the
transform dominates completely at full field.

## How long planning is worth, measured

Letting `FFTW_MEASURE` run unbounded was time given away. At 1799 x 1549 --
2.79 Mpoints, the same factor structure as the real padded case:

| planning | plan | DST | break-even |
|---|---|---|---|
| `estimate` | 0.1 s | 44.8 ms | -- |
| `measure`, 1 s | 1.2 s | 43.2 ms | 726 calls |
| `measure`, 5 s | 6.9 s | 39.4 ms | 1262 |
| **`measure`, 15 s** | 15.0 s | **32.8 ms** | 1252 |
| `measure`, 60 s | 49.0 s | 43.0 ms | 28 379 |
| `patient`, 60 s | 60.0 s (**withdrawn**) | 30.4 ms | -- |

**On a smooth size the whole spread is 1.47**, and fifteen seconds captures
almost all of it. Beyond that the budget is spent for nothing: 49 s of planning
returned 43.0 ms, worse than the 15 s plan. The default is now 15 s.

This compounds with the padding in a way worth noticing. The collapse of
`estimate` measured earlier -- 1228 ms against 97 -- belonged to the awkward
size. On a smooth one `estimate` is within 47 % of the best plan, so padding
does not merely divide the transform by 3.7: it also makes expensive planning
close to pointless.

One caveat on the method, and it turned out to matter. FFTW accumulates wisdom
**within the process**, so each plan benefits from its predecessors and the rows
are not independent -- the unlimited row built in 0.0 s because it reloaded.

The `patient` row is **withdrawn** on that ground. Rerun from an empty cache at
1800 square, `patient` with a 60 s limit did not finish its first plan in **nine
minutes**, wrote no wisdom, and produced nothing. Its 60.0 s in the table was
the benefit of the six rows before it. The time limit is a hint FFTW honours
only approximately -- it finishes the planning operation in progress -- and
under `patient` those operations are large enough that the hint means nothing.

So `patient` is not usable as a default: it cannot be bounded. `measure` with a
fifteen-second limit is the default, and `patient` stays available for a
deliberate offline planning run by someone willing to pay for it.

## The gate, at full field

```text
grid 3599 x 3099, 22 293 208 interior unknowns, 22 306 602 material points
kernel numba, preconditioner diagonal, backend fftw, 8 workers, setup 11.6 s

stiffness symmetry        9.963e-16     v.Kv  +8.005751e+18
preconditioner symmetry   4.309e-14     v.Mv  +1.178128e+02
conjugate gradient        29 iterations
adjoint dot product       4.445e-17     over 2 pairs
A 52.00 s   A^T 53.03 s   peak 1785 MB
```

**PASSED.** The adjoint holds at 4.4e-17, nine orders below the registered 1e-8,
at 22.3 million unknowns. Nothing guaranteed a priori that the adjoint of the
lifted primal would be consistent, and that was the whole point of the gate.

Twenty-nine iterations is exactly what 200 and 256 square gave with padding, so
mesh independence holds across three orders of magnitude in size. Symmetry and
positivity hold. Peak memory is 1.8 GB against the 14 GB allowed.

Wisdom pays as intended: the cold plan costs 249.2 s and writes the cache, a
warm build costs **8.0 s** and keeps the plan quality -- `M` at 830 ms against
874 -- so four minutes are spent once for this dataset and never again. `M` at
830 ms against 2560 unpadded confirms the 3.7 padding gain at the real size.

One estimate of mine was out by a factor of two and the reason is worth
recording. I predicted about 25 s from `29 x (M + K)`; the measurement is 52.
The gap is neither `M` nor `K` but the rest of the chain -- the dot products and
axpys of conjugate gradient over 178 MB vectors, plus the `B` and `B^T`
applications outside the loop. At this size the solver's own vector algebra
stops being negligible, and no extrapolation from 1024 square could have shown
it.

`T_A` over this milestone: **134 s to 52 s**, and qualified rather than hoped.

## Milestones 1A and 1B: the nonlinear loop with many local coefficients

Three increments of synthetic uniaxial Dirichlet loading, `Delta p` prescribed
from a coarse grid of coefficients under a bilinear partition of unity, the
global equilibrium solved by the existing Newton loop with a matrix-free
Jacobian and the spectral preconditioner from milestone 0.

**The coefficient count costs nothing.** Krylov totals over three increments:

| grid | 64 coeff | 256 | 1024 | 4096 |
|---|---|---|---|---|
| 256 | 168 | 175 | 161 | 156 |
| 512 | 166 | 175 | 166 | 158 |
| 1024 | 167 | 180 | 168 | -- |

Eight Newton iterations everywhere, at every size and every coefficient count.
Multiplying the coefficients by 64 changes neither Newton, nor Krylov, nor the
time. The architectural property holds: the field is assembled first and the
mechanics solved once, so coefficients cost interpolation.

**The preconditioner decides everything.** At 256 square, over the four
coefficient counts, preconditioned against not:

```text
168 Krylov,      56 s
145 649 Krylov, 6487 s
```

867 times fewer iterations and 115 times less time. Unpreconditioned, a single
case took nearly two hours.

**Mesh independence survives the nonlinearity**: 166 to 180 Krylov from 256 to
1024 square, the elastic behaviour unchanged. Time grows with the point count --
40, 115, 440 s -- so the full field extrapolates to roughly 78 minutes for three
increments.

The 1024-square, 4096-coefficient case was killed by the OOM killer, and the
cause was mine rather than the solver's: the partition of unity was
materialised as a dense `(pixels, patches^2)` matrix, which at that size is
1 048 576 x 4096 in float64, or 34 GB. The weights are separable, so the field
is now two small contractions -- agreeing with the dense route to 4.4e-16 and
summing to one to 2.2e-16.

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

## What follows: a nonlinear bench before the network

Milestone 0 qualified the *linear* operator and its adjoint. What is still
unknown is how the **global nonlinear loop** behaves once internal variables, an
algorithmic tangent, Newton-Krylov, many spatial coefficients and history all
arrive together. J2-Ludwik is the right bench for that -- not as a return to the
constitutive postulate this project exists to get past, but as an oracle whose
answer is known, whose plasticity localises, and whose tangent becomes spatially
variable under load.

```text
local coefficients -> d eps_p(x) -> sigma(x) -> R(u) -> global Newton-Krylov
```

with `R(u) = -B^T sigma` and `J(u) v = -B^T C_alg(x) B v`, matrix-free
throughout, no assembled tangent. The material parameters stay **homogeneous and
known**: locality comes from the plastic representation, not from an invented
map of `sigma_y(x)`, `K(x)`, `n(x)`, which would confound nonlinearity with
heterogeneous-property identification.

Two facts make this the right next step rather than a detour.

**J2 keeps symmetry.** The algorithmic tangent of associative J2 with isotropic
hardening is symmetric, unlike crystal plasticity's, so conjugate gradient stays
valid and everything qualified in this milestone survives unchanged.

**But the tangent contrast is far worse than the elastic one.** The Voigt/Reuss
bracket of 1.41 computed above for cubic 316L said crystalline *elasticity* will
not break the preconditioner. That does not transfer: in the plastic zone the
deviatoric tangent modulus collapses toward zero as hardening falls, so the
contrast is potentially unbounded. This is the real risk, and it is much sharper
than the number I reassured with earlier.

### The order, and one test that comes before all of it

**Milestone 1.0** -- does the homogeneous preconditioner survive a heterogeneous
tangent? Replace `C` by a synthetic `C_alg(x)`, a fraction of points softened by
a chosen factor, and measure `n_CG` against that fraction and that contrast. Ten
minutes, no Newton loop, and it separates "the preconditioner survives" from
"the loop works". If 29 becomes 300 at a contrast of 100, the preconditioner is
the next milestone and nothing else. If it stays under 60, 1A can be written
without reservation.

**Milestone 1A** -- J2-Ludwik forward, nonlinear matrix-free, on 256 to 1024
square with a clean synthetic Dirichlet loading rather than the DIC lifting,
whose repair is still outstanding. Report per increment: plastic fraction,
tangent contrast, Newton steps, Krylov per Newton, and the time split across
constitutive, Jacobian-vector, preconditioner and vector algebra. That last one
matters: at full field it already accounted for the factor of two between my
25 s estimate and the measured 52.

**Milestone 1B** -- local coefficients on that same known problem, on grids of
8x8, 16x16, 32x32 patches under a partition of unity. The property to verify is
that the coefficient count never multiplies the number of global solves:
assemble `q(x)` first, then solve once. Ten thousand coefficients must cost
local contractions, not ten thousand applications of `A`.

**Milestone 1C** -- the same at full field, with the measured DIC lifting.

**Then** the frozen CNN generator and the assembled-field projection.

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
