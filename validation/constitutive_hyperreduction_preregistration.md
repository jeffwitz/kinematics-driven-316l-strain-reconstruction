# Constitutive hyper-reduction on a reduced integration domain — preregistration

Registered before implementation. Section numbers refer to the specification
this campaign follows.

## What this is, and what it is not

```text
The present method is inspired by reduced-integration-domain hyper-reduction,
but retains the full displacement space and the full-field equilibrium
operator. Hyper-reduction is restricted to the nonlinear constitutive
correction around a globally evaluated elastic reference.
```

Ryckelynck's hyper-reduction reduces the displacement space and evaluates the
constitutive law only on a reduced integration domain. **We deliberately do not
reduce the displacement.** The equilibrium operator, qualified at full field
with an adjoint at 4.4e-17, stays exactly as it is. The only new degree of
freedom is *how many exact constitutive integrations we perform*:

```text
sigma   = sigma_n + C_0 : d eps + h
R_HR    = B^T [ sigma_n + C_0 : d eps + R_S(h_S) ]
```

The residual is still assembled over every point. Only the expensive part, `h`,
is sampled and reconstructed.

## The arithmetic that bounds the whole idea, before any code

The specification's own example takes 8 Newton iterations with a 10 % domain
and one certification: `8N -> 0.8N + N = 1.8N`, a gain of 4.4. That holds in
the cold regime.

**It does not hold in the regime training actually lives in.** We measured a
coefficient perturbation at 1 Newton, not 8. Then `1N -> 0.1N + 1N = 1.1N`: the
certification alone costs more than the entire solve it certifies, and
hyper-reduction is a **10 % loss**.

So the gain exists in exactly two places, and this is registered before the
work rather than discovered after it:

* the cold, multi-Newton regime, where it is worth about 4;
* the warm regime **only with amortised certification**, which the
  specification defers to `certification="periodic"` and does not qualify.

Any campaign result must be read against which regime it belongs to.

## Existing API this reuses, as required before coding

Nothing here builds a second infrastructure.

| piece | reused as |
|---|---|
| `PlaneStressMaterialBatch` protocol -- `evaluate(strain, time_increment, consistent_tangent)` returning `stress_in_plane_mpa` and `tangent_in_plane_mpa`, plus `commit()` / `revert()` | the constitutive contract; the adapter wraps it and never reimplements J2 |
| `PythonJ2PlaneStressBatch`, `MFrontNativePlaneStressBatch` | the two interchangeable behaviours behind that contract |
| `TwoSubcellDiagnostic2D` -- 2 material states per pixel, `material_point_count = 2 nx ny`, layout `(nx, ny, 2, 3)` | the point families the RID must respect; a RID must not silently mix the two subcells |
| `solve_fixed_plastic_increment_equilibrium`, `solve_two_state_dirichlet_plane_stress` | the Newton loops, unchanged |
| `FullFieldPlasticOperator.precondition`, the DST-I `B_0^-1`, the fused stencil | the mechanical backbone, **not to be touched** |
| the separable partition of unity from the local-coefficient bench | the reconstruction principle: never a dense `(N, N_RID)` matrix |

## A convention that must not be got wrong

`PythonJ2PlaneStressBatch` works in the **engineering (Voigt)** convention: its
tangent at zero strain equals `plane_stress_elasticity` exactly and differs from
the Kelvin stiffness by `mu` on the shear entry. The identification operator and
the spectral solver, by contrast, are Kelvin throughout.

`C_0` in the split must therefore be the engineering elastic stiffness, taken
from the material's own tangent at zero strain rather than assumed. Chaining the
two conventions without converting is what left the elastic lifting in a dozen
`scripts/*_p43.py` retaining 32 % of the interior residual.

## Which constitutive backend, settled by measurement

An earlier note of mine claimed MFront was two to six times *slower* than the
vectorised Python batch and withdrew it as a remedy. That was measured
**single-threaded**, which is not how this repository runs it -- its campaigns
set `mfront_threads: 4`.

At 80 000 points, same strains, same committed state:

| branch | Python | MFront t1 | t2 | t4 | t8 |
|---|---|---|---|---|---|
| elastic | **62.8 ms** | 398.6 | 234.5 | 171.4 | 121.9 |
| plastic | 387.1 ms | 776.0 | 444.0 | 263.0 | **209.1** |

MFront overtakes at four threads on the plastic branch and is **1.9x faster at
eight**, while Python keeps a factor of two on the elastic branch where the
per-point work is trivial and marshalling dominates. The crossover is set by
the cost of the local problem, which is also why MFront is unarguable for
crystal plasticity and marginal for elastic J2.

For the record, no trace exists of a Python constitutive law being benchmarked
against MFront in earlier campaigns. In
`p43_m100_backend_comparison_latest.json` the run labelled `python_condensed`
has backend `mfront-3d-condensed-plane-stress`: "python" names where the host
condensation loop runs, not the law. All three runs there are MFront, on SRIX,
within 10 % of each other.

## Registered gates, in order, each blocking the next

1. **The split is exact.** `sigma_n + C_0 : d eps + h` reproduces the original
   `sigma` to rounding, full field.
2. **100 % RID is the identity.** With `S = Omega` and `R_S = I`, the
   hyper-reduced residual and Jacobian match the full ones to **1e-12 or
   better**. The reduction campaign does not start until this passes.
3. **Elastic invariance.** Where the response is elastic, `h = 0` and
   `Delta C = 0`, so the answer must be *independent of the RID size* -- 2 % and
   100 % must agree exactly.
4. **The reconstruction has an exact transpose.** `R_S` is not symmetric, so
   `apply_transpose` is implemented separately and the dot product tested below
   1e-8, targeting 1e-10 on a small grid.
5. **The Jacobian is the derivative of the residual it goes with**, verified by
   central differences over a step sweep, not only by a dot product.
6. **The committed state is immutable during Newton.** Every trial integrates
   from `z_n`, line searches included.
7. **Nothing is committed uncertified.** A converged hyper-reduced candidate is
   integrated exactly at full field; the reported residual is always
   `|R_full(u_HR)|`, never `|R_HR|`, and the state committed is the exact one.

## Registered falsifiers

* The gain does not come from fewer constitutive integrations. The counter
  `n_constitutive_point_updates` must show it directly; a speedup traceable to
  anything else is not this method working.
* The gradient with respect to the local coefficients is biased. `cos(g_HR,
  g_FOM)` and the relative error decide this, and they matter more than the
  displacement error, since training consumes the gradient and not the field.
* No knee exists in the accuracy-against-cost curve. If error grows as fast as
  cost falls, there is no operating point to choose and the method is refuted
  for this problem.

## Scope of the first commit

The exact split, the adapter, a structured RID with separable reconstruction,
the algebraic identity tests, and the J2 campaign at 256 and 512 square. No
POD, no adaptivity, no crystal plasticity, and 1024 square only once the
accuracy-against-cost curve at 512 is usable.

J2-Ludwik remains a numerical bench. It is not a return to the constitutive
postulate the reconstruction exists to get past.
