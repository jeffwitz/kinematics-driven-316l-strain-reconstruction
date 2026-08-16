# Reusing the full-Dirichlet solver for the plastic inverse

```{admonition} The full-Dirichlet solver is not a new development
:class: important

The DIC-driven plastic reconstruction needs a mechanical operator that scales to
`3599 x 3099`. That operator is **this one**. What has to be written is the
matrix-free plastic inverse `A / A^T` on top of it, not a new boundary
treatment, not a new spectral solve, and not a new preconditioner.
```

This page exists because the reconstruction work drafted two consecutive plans
around rebuilding a Dirichlet solver that has been here, documented, all along.
The reconstruction notes under `validation/` did not link to this section, so
the reader of one had no reason to look at the other.

## What is already provided

The splitting, from {doc}`full_dirichlet_formulation`:

```text
u = u* + u^f,      u*|_boundary = u_DIC,      u^f|_boundary = 0
```

`u*` is a discrete harmonic extension of the measured boundary displacements
(`spectral2d/boundary.py::HarmonicDirichletExtension2D`), so the transform acts
on the homogeneous fluctuation and **the problem is never treated as periodic**.

The pipeline, from {doc}`solver_pipeline`: local kinematics give `sigma` and
`C_alg`, the residual is `R = -sum B^T sigma`, and a matrix-free Newton--GMRES
uses `J v = -sum B^T C_alg B v`. The DST-I basis, compatible with `u^f = 0` on
all four edges, applies the reference inverse `B_0^-1` **as a preconditioner
only**. `B_0` is a Gelebart-type reference operator and deliberately not the
exact inverse of the coupled isotropic stiffness. No global stiffness is
assembled anywhere.

## What the plastic inverse adds

Only the eigenstrain path and its adjoint:

```text
d eps_p
   |  mechanical source
   v
equilibrium, matrix-free           J v = -sum B^T C B v
   |  preconditioned by B_0^-1 through DST-I
   v
u^f
   |
   v
d eps = A d eps_p
   |
   v
E_DIC
```

and, coming back,

```text
E_DIC  ->  A^T  ->  DP_H^T  ->  Phi_local^T  ->  grad_a E
```

Two differences from the production pipeline are worth stating. This solve is
**linear** -- `A` is the elastic response to an eigenstrain, not the
constitutive Newton loop -- so it is one preconditioned Krylov solve with no
Newton iteration. And because `B_0` is a reference operator rather than the
exact inverse, that linear solve still iterates; the iteration count, not the
transform cost, is what sets `T_A`.

## The rule that makes it scale

```{admonition} Never apply the solver once per mode
:class: warning

Building `A Phi` column by column costs one solve per mode, which is what made
an earlier estimate read 32 minutes per gradient step. Assemble the plastic
field first,

    v = sum_jk a_jk w_j phi_jk   ->   q = P_H(v)   ->   A q

then obtain the gradient of **every** coefficient from a single `A^T`:

    p = A^T W (A q - g)          ->   dJ/da_jk = <DP_H(w_j phi_jk), p>

The number of local coefficients then costs local contractions, not global
solves.
```

The representation is local -- modes and coefficients per subdomain, joined by a
partition of unity. The mechanics stays global. Windows carrying DIC data on
their own contour are useful training material and never proof: the boundary
kinematics of a window already contains the effect of everything outside it, and
independently solved windows guarantee nothing about `B^T sigma = 0` over the
whole domain.

## Where the rest is written

`validation/full_field_operator_gate.md` holds the registered milestone,
its tasks and its acceptance criteria, and
`validation/dic_driven_plastic_identification.md` is the cold-restart document
for the reconstruction as a whole.
