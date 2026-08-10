# Spectral Newton–GMRES solver pipeline

This page connects the three ingredients that are documented separately in
the surrounding chapters: the full-Dirichlet displacement decomposition, the
matrix-free constitutive Jacobian, and the DST-I reference preconditioner.

The central distinction is:

> The spectral solver does not replace local kinematics or constitutive
> integration. It replaces explicit construction and inversion of the global
> stiffness matrix.

The residual is evaluated from the actual local stresses. The Jacobian action
is evaluated from the actual local algorithmic tangents. The DST-I is used only
inside GMRES to apply the inverse of the homogeneous elastic reference
operator $B_0$.

## One global increment

For an increment from $n$ to $n+1$, the measured boundary displacement is
extended into the domain and the displacement is written as

```{math}
u = u^ast + u^f,
\qquad u^\ast|_{\partial\Omega}=u^{\rm DIC},
\qquad u^f|_{\partial\Omega}=0.
```

Only the interior fluctuation degrees of freedom are unknown to the global
solver. At each Newton iterate, the local strain operator $B$ maps those
unknowns to element/quadrature-point strains. The configured constitutive
adapter then integrates the material response, including the local structural
plane-stress closure when selected. For TET2, the material layer carries two
local histories per pixel; other discretisations expose their configured
number of local states.

The complete loop is:

```{graphviz}
digraph newton_gmres_pipeline {
  rankdir=TB;
  node [shape=box, style="rounded,filled", fillcolor="#eef4fb"];
  edge [color="#31465a"];

  inc [label="Increment n -> n+1\nDIC displacement data"];
  split [label="Full-Dirichlet split\nu = u* + uf\nuf = 0 on boundary"];
  newton [label="Newton iterate k\ncurrent interior displacement u"];
  kin [label="Local kinematics\nepsilon = B u"];
  mat [label="MFront / constitutive adapter\nsigma, C_alg\nlocal plane stress if selected"];
  res [label="Actual residual\nR(u) = -sum B^T sigma", fillcolor="#fff4df"];
  jv [label="Matrix-free Jacobian action\nJ v = -sum B^T C_alg B v", fillcolor="#fff4df"];
  gm [label="GMRES\nsolve J du = -R"];
  kry [label="Krylov residual r"];
  dst [label="DST-I -> divide by B0 symbol\n-> inverse DST-I\nB0^-1 r", fillcolor="#e8f5e9"];
  upd [label="Update u <- u + du"];
  test [shape=diamond, style="filled", fillcolor="#fff4df", label="Global residual\nconverged?"];
  commit [label="Reintegrate verified trial\nthen commit", fillcolor="#e8f5e9"];

  inc -> split -> newton -> kin -> mat;
  mat -> res;
  mat -> jv;
  res -> gm [label="-R"];
  jv -> gm [label="operator callback"];
  gm -> kry;
  kry -> dst [label="preconditioner"];
  dst -> gm [label="correction"];
  gm -> upd [label="du"];
  upd -> test;
  test -> commit [label="yes"];
  test -> newton [label="no"];
}
```

The constitutive evaluation and the Jacobian callback use the same current
trial state. A rejected line-search or Newton trial is reverted to the last
globally committed state; convergence is followed by an independent verified
integration before commit.

## Where the spectral method acts

The distinction from a conventional sparse FEM solve is localised to the
linear Krylov solve:

| Operation | Sparse FEM implementation | Full-Dirichlet spectral implementation |
|---|---|---|
| Local strain | element operator $B$ | element operator $B$ |
| Constitutive response | material integration | material integration |
| Nonlinear residual | assembled or matrix-free internal force | matrix-free internal force $sum B^T\sigma$ |
| Jacobian | assembled global $K$ or matrix-free action | matrix-free action $sum B^T C_{\rm alg} Bv$ |
| Linear correction | sparse factorisation or iterative preconditioner | GMRES with $B_0^{-1}$ applied by DST-I |
| Boundary treatment | constrained global system | zero-boundary fluctuation, diagonal DST-I modes |

There is no global plastic stiffness matrix assembled for the spectral path.
The local material tangent still controls the exact Jacobian action. The
homogeneous reference operator $B_0$ is deliberately simpler than that
Jacobian; it is selected because its inverse is inexpensive and effective as
a preconditioner.

## The DST-I preconditioner

For a Krylov residual $r$, the preconditioner performs exactly the following
operations:

```text
r in physical space
    -> orthonormal DST-I in x and y
    -> divide each displacement component by its B0 modal symbol
    -> inverse DST-I
    -> approximate elastic correction B0^-1 r
```

This is not a second constitutive solve and it is not the calculation of the
nonlinear residual. It is the response of a homogeneous elastic reference
problem used to rescale Krylov directions. The word *predictor* can be useful
as a physical interpretation—$B_0^{-1}$ predicts the correction of the
reference elastic medium—but its solver role is the **spectral elastic
preconditioner**.

The transform is a DST-I because the fluctuation vanishes on the boundary. It
is not a periodic DFT/FFT and does not invoke a periodic Lippmann–Schwinger
formulation. The modal diagonalisation applies only to $B_0$, not to the
heterogeneous plastic Jacobian.

## Minimal algorithm

The following pseudocode shows the ownership of each operation. It omits
line-search details, batching and diagnostics, but preserves the numerical
separation between the true material path and the reference preconditioner.

```python
for increment in increments:
    u = displacement_predictor(increment)

    for newton_iteration in range(max_newton):
        strain = B(u)
        stress, C_alg = material.evaluate(strain)
        residual = BT(stress)

        if converged(residual):
            material.reintegrate_from_committed_state(u)
            material.commit()
            break

        def jacobian_action(v):
            return -BT(C_alg * B(v))

        def elastic_preconditioner(r):
            modes = DST_I(r)
            modes /= B0_modal_symbol
            return inverse_DST_I(modes)

        du = GMRES(
            operator=jacobian_action,
            rhs=-residual,
            preconditioner=elastic_preconditioner,
        )
        u += du
```

The signs in the pseudocode follow the residual convention used by the
matrix-free implementation. The exact callback and transaction contract is
specified in {doc}`../../reference/numerics/newton_gmres_contract`; the
full-Dirichlet kinematics are derived in
{doc}`full_dirichlet_formulation`; and the modal $B_0^{-1}$ action is derived
in {doc}`dtt_green_operator`.
