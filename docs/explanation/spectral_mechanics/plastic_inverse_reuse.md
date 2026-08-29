# Full-field mechanical operator and adjoint

This page explains how the full-Dirichlet spectral mechanics can be reused for
inverse problems. It is a method-level capability, not a claim that a P43
plastic field or material has been uniquely identified.

## The mechanical operator

The full-Dirichlet split is

```text
u = u* + uf,      u*|boundary = u_DIC,      uf|boundary = 0
```

`u*` is a discrete harmonic extension of measured boundary displacement. The
fluctuation is therefore homogeneous at the boundary and the transform is
never used to make the physical problem periodic.

The nonlinear forward retains the actual local constitutive response:

```text
local kinematics -> stress and C_alg
                 -> R = -sum B^T stress
                 -> Jv = -sum B^T C_alg Bv
```

The DST-I inverse of the reference $B_0$ is only the Krylov preconditioner;
the global constitutive tangent is not assembled or diagonalised in Fourier
space. See {doc}`full_dirichlet_formulation` and {doc}`solver_pipeline`.

For a small perturbation of plastic/eigenstrain, the linearised field map is

```{math}
A:\delta\varepsilon_p\longmapsto\delta y,
\qquad
A^T:g_y\longmapsto g_{\varepsilon_p}.
```

Here $y$ can be a displacement, strain or declared observed field after the
appropriate observation map. The `A` solve is linear around the registered
reference state; it is not a replacement for the nonlinear constitutive loop.

## Why the adjoint matters

Suppose a local field is expanded as

```{math}
q=\sum_i a_i\phi_i.
```

A naive implementation applies $A$ once per basis function. The scalable route
assembles $q$, applies $Aq$ once, then applies $A^T$ once to the objective dual
and obtains all coefficient gradients by local contractions,

```{math}
p=A^Tg,
\qquad
\frac{\partial J}{\partial a_i}=\langle\phi_i,p\rangle
```

with the declared partition-of-unity and chain-rule factors included. A large
number of local coefficients therefore need not imply one global mechanical
solve per coefficient. The coefficient count affects local assembly and
contractions, while the global solve count stays tied to the forward/transpose
actions.

## Registered full-field gate

The full-field operator gate reports:

```text
grid                 3599 x 3099
interior unknowns    22,293,208
A                    approximately 52 s
A^T                  approximately 53 s
peak memory          approximately 1.8 GB
adjoint discrepancy   4.445e-17
```

This is a transpose-consistency and full-field feasibility result for the
registered homogeneous-elasticity configuration. Timings are machine- and
planning-dependent; they are not universal performance promises. The complete
acceptance record is `validation/full_field_operator_gate.md`.

## Adjoint levels must not be conflated

The repository contains related but distinct constructions:

1. the qualified full-field linear/eigenstrain operator $A^T$ described here;
2. mechanical adjoints of selected converged inverse problems;
3. a sequential history adjoint in the causal TANN exploration.

The latter two require their own state, objective and trajectory contracts.
The full-field gate does not establish a generic analytic SRIX parameter
adjoint, nor does it validate a trained TANN constitutive model.

## What this enables

The operator and adjoint provide ingredients for high-dimensional field
inversion, gradient-based FEMU, material-field optimisation and future
topology optimisation. They also make it possible to test many local
coefficients without repeating a complete global solve for each one. These are
reusable numerical ingredients, not completed optimisation products.

Local windows carrying their own DIC boundary data remain training or diagnostic
material, not proof of global equilibrium: a window boundary already contains
the influence of material outside the window.

The reconstruction-specific historical reports remain in
`validation/dic_driven_plastic_identification.md`; this page exposes the
operator as a general methodological capability.
