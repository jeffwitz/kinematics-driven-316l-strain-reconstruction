# Milestone 3 — the free tensor plastic increment, local coefficients, no network

Registered before any run. Thresholds frozen. Negative results kept.

## Why the previous milestone is not the right family

Milestone 2 (`07c1fc9`) qualified the inverse **plumbing** and is kept: exact
adjoint, verified gradient, warm start at one Newton, and an identifiable local
parameterisation recovered to `2.1e-4`. None of that is retracted.

What is retracted is the *representation*. That chain prescribes a **scalar**
`Delta p(x)` and lets `DrivenJ2PlaneStressBatch` impose the associated J2 flow
direction, so the plastic increment is confined to

```text
Delta eps^p(x) = Delta p(x) n_J2(sigma(x)),      Delta p >= 0.
```

One degree of freedom per point, and the direction dictated by the model.
Varying only the amplitude along a fixed isotropic direction was already found
insufficient to absorb the DIC kinematic defect, and enriching that scalar with
polynomial modes (`q > 1`) would add spatial richness to the **amplitude** while
leaving the **tensor direction** just as constrained. So the enrichment repair
is dropped and the family changes instead.

## The family under test

Three Kelvin coefficients per patch node, assembled by the same partition of
unity, whose degree-zero conditioning was measured at **3.4** with no null
direction:

```text
a_j = [a_j,xx , a_j,yy , a_j,xy^K]^T,        v(x) = sum_j w_j(x) a_j
```

with `v` read as the Kelvin triple `[de_xx, de_yy, sqrt(2) de_xy]`. At
`8 x 8 x 3` that is **192** local variables against 64 in the scalar milestone.
No network, no `q > 1`, no local J2.

Out of plane follows from plastic incompressibility, `tr_3 = 0`, hence
`de_zz = -(de_xx + de_yy)`. **No projection enforces this and none should**:
demanding a vanishing *in-plane* trace would force `de_zz = 0`, a plane-strain
plasticity this specimen does not have.

## Admissibility after assembly, which is the whole point

```text
Delta eps^p,K(x) = P_{H(sigma_pred(x))} [ v(x) ],     H(sigma) = { z : sigma^T z >= 0 }
```

reusing the projection already qualified in branch D:

```text
P_H(v) = v + ReLU(-sigma^T v) / (sigma^T sigma) * sigma
```

so that only the genuinely anti-dissipative component is removed and **every
direction tangent to the stress survives**. Because `sigma_zz = 0` in plane
stress, the in-plane Kelvin dot product equals the full three-dimensional
`sigma : eps^p` exactly, so this half-space is the complete thermodynamic
condition and not an in-plane shadow of one.

The order is not negotiable and is what makes this branch D:

```text
P_H( sum_j w_j a_j )        and never        sum_j w_j P_H(a_j).
```

Projecting mode by mode and then blending gives a different field -- by up to
62 % on the same coefficients.

> **Correction, recorded after the run and not before it.** The rationale
> originally written here was that a mode-wise blend could sum to an
> *inadmissible* field. That is false: `H_sigma` is a convex cone, so a
> partition of unity's non-negative blend of admissible contributions is always
> admissible, and 200 random coefficient sets never produced a violation. The
> ordering still matters, for a different reason: mode-wise projection clips
> each contribution in isolation, which shrinks the reachable family and ties
> the result to an arbitrary decomposition into modes. No threshold changed.

`sigma_pred` is **frozen** before the run and does not depend on `a`: a uniform
uniaxial elastic predictor plus the elastic response to an independently seeded
smooth eigenstrain, so the half-space varies in space without leaking the truth
being recovered.

## Mechanics

The already-qualified matrix-free tensor operator, not the scalar `DrivenJ2`:

```text
sigma = C^ps : (eps - eps^p),      B^T sigma = 0,      A: Delta eps^p,K -> u
```

`TensorPlasticObservabilityOperator` supplies `A` and an exact `A^T`. The
mechanics is therefore **linear** in the plastic field and `P_H` is the only
nonlinearity in the chain, which removes the Newton loop entirely.

## Gates, in order

**Gate 1 — the transpose.** `<A x, y> == <x, A^T y>`, relative `<= 1e-10`.

**Gate 2 — the gradient.** Central differences against the adjoint over a
four-decade step sweep, best relative error `<= 1e-5` **and** V-shaped.
`P_H` is piecewise linear, so the base point is chosen away from the kink and
the sweep is rejected if any point changes activity across the perturbation;
that check is part of the gate, not a convenience.

**Gate 3 — the spectrum of `du/da`.** Reported in full. Registered:
`>= 90 %` of the 192 directions above `1e-6` of the leading singular value,
mirroring the clean degree-zero result. Anything less means the tensor
parameterisation carries its own degeneracy and must be fixed before use.

**Gate 4 — twin recovery of the tensor field.** Generate `u_obs` from a known
smooth, strictly dissipative `a_true` with three independent components, start
elsewhere, recover. The registered quantity is the **plastic-gauge** relative
error, never the coefficient error:

```text
|| Delta eps^p - Delta eps^p_true ||_Gp  /  || Delta eps^p_true ||_Gp   <=   5 %
```

with `Gp = (2/3) [[2,1,0],[1,2,0],[0,0,1]]`, the gauge whose norm is the
equivalent plastic strain. Using a plain Euclidean norm here would misweight
shear by a factor of three and is the mistake this line exists to prevent.

**Gate 5 — admissibility.** Zero points with `sigma_pred^T Delta eps^p < 0`
after `P_H`, and the active fraction of the projection reported at every
iterate.

**Gate 6 — cost.** Reported, no threshold: the mechanics is now one linear solve
per evaluation rather than a Newton loop, so the scalar milestone's Newton
counts do not transfer.

**Gate 7 — the family comparison, synthetic.** The decisive test is `E_tensor`
against `E_{Delta p + J2}` on real DIC and that waits for the elastic lifting
repair. A controlled version is available now and is registered here: generate
`u_obs` from a tensor field that is **not** J2-parallel, then fit it with both
families through *identical* mechanics and *identical* objective, the scalar arm
restricted to `Delta eps^p = Delta p(x) n_J2(sigma_pred(x))` with `Delta p >= 0`.

Registered expectation, and a falsifier if it fails: the tensor family reaches a
**materially lower** objective than the scalar one, by at least one decade. If
the scalar family matches the tensor family on data it structurally cannot
represent, the twin generator is not exercising the extra freedom and the
experiment is uninformative rather than favourable.

## Registered falsifiers

* The gradient sweep plateaus instead of showing a V.
* Gate 4 reaches a low objective with a plastic-gauge error still large: the
  operator `A` is surjective, so fitting a displacement is not evidence about a
  field, and this is the outcome that would say so.
* `P_H` is active on a large fraction of points at the optimum, which would mean
  the descent is driven by the projection rather than by the data.
* The tensor family and the scalar family reach the same objective on gate 7.

## Out of scope, deliberately

No network. No `q > 1`. No crystal plasticity. No reduced integration domain —
its reopening condition is registered elsewhere and nothing here touches it. No
claim about 316L: the twin is generated by the exact forward model being fitted,
which isolates the inverse from model error and is not a statement about real
data.

## After this

Repair the elastic lifting with an asserted Kelvin/engineering/Voigt
conversion, run gate 7 against the real DIC, and only then introduce the
generator — which must learn a **local tensor family** whose admissible
realisation is selected by the mechanics and the dissipation projection, not a
map of J2-associated `Delta p`.
