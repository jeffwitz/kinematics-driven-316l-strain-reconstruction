# TET2 operator contract

TET2 exposes two strain samples per pixel and the exact weighted adjoint
divergence. The pixel area and quadrature weights are

```{math}
A_e=h_xh_y,
\qquad w_1=w_2=\tfrac12,
\qquad w_qA_e=\tfrac12h_xh_y.
```

Thus each geometric triangle has area $h_xh_y/2$; the weights are not
additional geometric areas. The operator must
satisfy, for arbitrary zero-boundary `v` and sample stress `s`,

```{math}
\langle s,Bv\rangle=-\langle\operatorname{div}(s),v\rangle.
```

The production state count is `2 * nx * ny`. The operator does not add
hourglass stiffness, spectral filtering or constitutive averaging.

The one-point operator is retained as a negative stability witness; it is not
the TET2 oracle.
