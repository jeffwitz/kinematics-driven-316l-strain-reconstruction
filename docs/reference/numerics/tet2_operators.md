# TET2 operator contract

TET2 exposes two strain samples per pixel and the exact weighted adjoint
divergence. Each triangle has weight \(1/2\) and area \(h_xh_y\). It must
satisfy, for arbitrary zero-boundary `v` and sample stress `s`,

\[
\langle s,Bv\rangle=-\langle\operatorname{div}(s),v\rangle.
\]

The production state count is `2 * nx * ny`. The operator does not add
hourglass stiffness, spectral filtering or constitutive averaging.

The one-point operator is retained as a negative stability witness; it is not
the TET2 oracle.
