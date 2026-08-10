# SRIX semismooth Jacobian at zero slip

## Production convention

The Forest–Rubin SRIX law contains `abs(dg)` in its isotropic hardening and
backstrain updates. The production residual and state update retain this exact
expression. Only the local Newton linearisation needs a convention at the
nondifferentiable point `dg = 0`.

The canonical convention is

```text
d|dg|/ddg = +1  if dg > 0
            -1  if dg < 0
             0  if dg = 0.
```

This is not a constitutive smoothing and is not exposed as a user option.

## Why zero is a valid generalized Jacobian element

For `f(x) = |x|`, the classical derivative does not exist at zero. Its
Bouligand limiting derivatives are

$$
\partial_B f(0) = \{-1,+1\},
$$

while the Clarke generalized derivative is their convex hull,

$$
\partial_C f(0) = \operatorname{conv}(\partial_B f(0))=[-1,+1].
$$

The implementation selects the symmetric Clarke element `0`. The historical
expression `x > 0 ? 1 : -1` selected `-1` at zero only as a consequence of its
branch syntax; that choice has no constitutive meaning.

The distinction matters. Selecting zero changes only the generalized Jacobian
passed to the local Newton solve. The value of `abs(dg)`, the residual, and the
committed state update are unchanged.

## P43 M200 qualification

The test case is P43 M200 EBSD, crop `[1520:1720] × [985:1185]`, eight
increments, the StructuralPlaneStress backend, four MFront threads, and
single-threaded BLAS/FFTW. The numbers below are single qualification runs;
they document work and convergence, not a repeated performance benchmark.

| Convention | Newton | Newton per increment | Substepped points | Composite-FD trajectories | Elapsed |
|---|---:|---|---:|---:|---:|
| historical `dg=0 → -1` | 58 | `[6,6,7,7,7,7,8,10]` | 978 | 5868 | 305.02 s |
| canonical `dg=0 → 0` | 56 | `[6,6,7,7,7,7,8,8]` | 0 | 0 | 130.50 s |

The archived local replay contained 380 isolated failed full-step
integrations. The historical convention failed all 380; the zero convention
rescued all 380. The local tolerance sweep also showed that, when both
conventions converge, their solutions approach the same root as the tolerance
is tightened (the difference fell from about `1e-7` to `1e-13`).

By contrast, a compact `|dg|` regularisation with `delta = 1e-5` changed the
function itself and retained a field difference of order `1e-4`. It is therefore
not the production solution.

## Why the old and new full-field fields differ

The zero convention removes the need for selective substepping. The historical
and canonical runs therefore use different incremental paths. A controlled
same-partition comparison agrees to approximately `6e-11`, whereas forcing the
canonical convention to use the one-step path produces differences of order
`1e-2` in local history variables. The previously observed full-field
differences of order `1e-4` are consequently attributed to the different
sub-increment partitions, not to a change in the constitutive root.

## References

- R. Mifflin, “An Algorithm for Constrained Optimization with Semismooth
  Functions,” *Mathematics of Operations Research* 2(2), 191–207 (1977),
  [doi:10.1287/moor.2.2.191](https://doi.org/10.1287/moor.2.2.191).
- L. Qi and J. Sun, “A nonsmooth version of Newton’s method,” *Mathematical
  Programming* 58, 353–367 (1993),
  [doi:10.1007/BF01581275](https://doi.org/10.1007/BF01581275).
- J. Sun and D. Sun, “Semismooth Matrix-Valued Functions,” *Mathematics of
  Operations Research* 27(1), 150–169 (2002),
  [doi:10.1287/moor.27.1.150.342](https://doi.org/10.1287/moor.27.1.150.342).
