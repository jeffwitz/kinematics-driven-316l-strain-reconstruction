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

## Numerical qualification

The convention has been verified on the qualified P43 M200 EBSD workflow. The
same SRIX residual and state update converge to the same local constitutive root
when the local tolerance is tightened; the selected generalized Jacobian only
changes the Newton path. The production implementation therefore has no
calibration knob for this choice.

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
