# Mechanically generated adaptive reduced basis — preregistration

Registered before any run. Thresholds, falsifiers and the negative verdict are
fixed here and are not to be moved after seeing results.

## Purpose

Every reduction attempted so far asked a fixed low-dimensional object to carry
the plastic field: global POD, POD per Laplacian band, a convolutional
autoencoder, a CROM neural field. All failed, and the shared premise is now
identified as the fault. **The plastic field need not lie on a global
low-dimensional manifold.** What must be low-dimensional is the space of
*admissible plastic corrections around the current mechanical state*.

So the object to reduce changes:

```text
refuted     eps_p_n = Phi a_n            one fixed Phi(x) for every state
registered  eps_p_n = Phi_theta(S_n) a_n  a basis the current state generates
```

The inverse problem still solves for `r` numbers per increment. The basis they
weight is free to move a band, reorient it, relocalise it, or follow the
microstructure, because it is regenerated from the mechanical state at every
increment.

## The formulation

At increment `n`, with `eps_p_{n-1}` known and the new DIC boundary conditions
applied, the predictor carries no new plasticity:

```text
eps_hat_n = eps_n^el + A eps_p_{n-1}
g_n       = eps_n^DIC - eps_hat_n
```

A shared generator sees the *predictor* state, never the interior DIC target:

```text
S_n(x) = { sigma_n^K(x), eps_n^K(x), history_{n-1}(x) }
F_theta(S_n) -> Phi_n = [phi_{n,1} ... phi_{n,r}]
d eps_p_n(x) = sum_k a_{n,k} phi_{n,k}^K(x)
```

`a_n in R^r` are the only unknowns of the inverse. Ranks swept: 2, 4, 8, 16, 32.

## Two protections this campaign adds to the specification

### The coefficients must not be fitted against the test data

`a_n` is global, so a `g_n` covering the whole field lets the held-out region
enter the fit. The leak is small at `r = 8` and is not small at `r = 32`, and
closing it costs nothing.

**Registered:** the reduced least squares is masked to the training region; the
reported `E_DIC` is measured strictly inside the held-out regions, which are
`HOLDOUT_REGIONS` as already used by every morphology benchmark. Any figure
computed on the fitting region is labelled a training figure and never compared
against a threshold.

### The dissipation cone in `R^r` is expected to be trivial

`C a >= 0` is the intersection of millions of half-spaces in `R^8`, which is
generically `{0}`. This is not a conjecture: the residual-driven Krylov subspace
admitted only `q = 0` at ranks 8, 32 and 128 alike. Reducing the dimension does
not change that geometry.

**Registered:** two mode parameterisations are run, identical in every other
respect.

```text
free        phi_k = q_1 E_1 + q_2 E_2 + q_3 E_3        trace-free, cone enforced by QP
aligned     phi_k = m_k(x) N(sigma_bar(x)),  m_k >= 0  dissipation by construction
```

with `N` the normalised deviatoric stress direction, so `D_g = m_k |sigma_dev|`
and the millions of constraints collapse to `r` sign conditions. Trace-freedom
holds in both.

The EBSD arm replaces `N` by the crystallography rather than adding an input
channel:

```text
crystal     phi_k = sum_alpha gamma_k^alpha(x) sign(tau^alpha) P^alpha(x),  gamma >= 0
```

`P^alpha` are the symmetric Schmid tensors from the verified EBSD orientations,
`tau^alpha` the resolved shear stresses. Dissipation is `sum gamma |tau| >= 0`
automatically and `tr P^alpha = 0` since `n . d = 0`. This makes the section 9
ablation structural: same architecture, same rank, same optimiser, and the
comparison asks whether slip-system directions beat an isotropic normal.

## What is measured

The demonstrator is P43 at 100x100, where `TensorPlasticObservabilityOperator`
already supplies `matvec`, `rmatvec` and `kelvin_response` in Kelvin throughout.
Three objects on **exactly the same increments**:

```text
free plastic field      the unconstrained inverse, an upper bound on fit
fixed POD / Krylov      the refuted baseline, rebuilt at the same ranks
adaptive Phi_theta(S)   this campaign
```

`E_DIC(r)` is the held-out relative error, `r = 2, 4, 8, 16, 32`.

## Outcome of the registered criteria

**Criterion 1 is unreachable by construction and the campaign is inconclusive on
its own terms.** The free plastic field, unconstrained and fitted exactly on the
training region, still leaves 0.6028 of the elastic defect inside the held-out
square. No basis can go below that, so `E_DIC(16) <= 0.10` was never attainable.
The threshold is not moved: it is recorded as unreachable, established by a
control involving none of the methods under test. Designing a spatial holdout
was the error, since it measures extrapolation into a hole that equilibrium and
the surrounding data do not determine. Results in
`adaptive_reduced_basis_first_rung.md`; a temporal holdout needs its own
preregistration.

## Registered acceptance criteria

A reduction is claimed only if **all four** hold.

1. `E_DIC(r)` falls monotonically over the sweep and reaches **`E_DIC(16) <=
   0.10`** on held-out data.
2. The adaptive basis beats the fixed basis of the same rank by at least a
   factor of two at `r = 8` and `r = 16`. A fixed basis that matches it means
   the state dependence bought nothing.
3. The midpoint dissipation `D_{n,g} = ((sigma_{n-1} + sigma_n)/2) . d eps_p_n`
   is non-negative at every material point, verified after convergence and not
   only enforced during it.
4. `cond(A Phi)` stays finite and no retained mode sits in the numerical kernel
   of `A`: the smallest singular value of `A Phi` is above `1e-8` of the
   largest. A mode that is invisible to the mechanics does not count toward `r`.

## Registered falsifiers

* **`E_DIC(r)` flat near zero from `r = 1`.** The generator has smuggled the
  answer into the basis. This is a failure, not a success, and the held-out
  measurement is what exposes it.
* **`E_DIC(r)` flat and high across the whole sweep.** The adaptive basis buys
  nothing over the fixed one and the reduction hypothesis is refuted in its
  strongest available form.
* **The aligned and crystal cones are also trivial**, admitting only `a = 0`.
  That would place the obstruction in the dissipation requirement itself rather
  than in the choice of basis, and it must be reported as such.
* **The Gram matrix collapses**, `G = Phi^T G_p Phi` far from `I` with modes
  duplicating each other, so the effective rank is below the nominal one. The
  reported `r` is then the effective rank, never the nominal.

## What the negative verdict would mean

Only a failure of the *adaptive, mechanically generated* basis licenses the
statement that this dataset does not admit a reduced plastic description. The
earlier POD, CAE, INR and inpainting results are retained as negative controls
on the fixed-basis premise and are not evidence about this one.

## Scope

P43 100x100 only. Full field is out of scope here and is conditional on the
demonstrator succeeding; it needs `A Phi` and `A^T v` fast, which is `r`
operator applications per increment and is the reason `r` must stay small.

No campaign already archived is replayed. Measured DIC boundary conditions enter
the solver untouched.
