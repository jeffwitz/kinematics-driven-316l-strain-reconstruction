# Plane-stress numerical contract

**Category: Reference.**

The global finite-element problem has two displacement degrees of freedom per
node. Its strain vector is
$[\varepsilon_{11},\varepsilon_{22},\gamma_{12}]$ and its stress vector is
$[\sigma_{11},\sigma_{22},\sigma_{12}]$.

For native MFront plane stress, MFront solves its axial constraint internally
and returns an already condensed tangent. For the 3D adapter, the local
transverse unknowns are solved and the tangent is condensed before the global
solver receives it.

Complete $3\times3$ tensors are output products. They do not enter the global
residual, tangent or Newton correction.
