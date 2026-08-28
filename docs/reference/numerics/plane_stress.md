# Plane-stress numerical contract

**Mode:** reference  
**Domain:** plane-stress

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

## Three-traction contract

With Kelvin components split into in-plane (a=(xx,yy,xy)) and transverse
(b=(zz,xz,yz)), the local closure solves

$$\varepsilon_b=(\varepsilon_{zz},\varepsilon_{xz},\varepsilon_{yz}),
\qquad \sigma_b=(\sigma_{zz},\sigma_{xz},\sigma_{yz})=0.$$

For a three-dimensional tangent partitioned into (aa,ab,ba,bb), the
in-plane algorithmic tangent returned to the global solver is

$$C^{PS}=C_{aa}-C_{ab}C_{bb}^{-1}C_{ba}.$$

The repository provides native MFront plane stress, external 3-D condensation,
structural plane-stress MFront and native SRIX nested/coupled closures. They
share this contract but differ in where the local Newton solve is performed;
the generic MFront bridge naturally uses the nested strategy.

The full structural derivation is in {doc}`mfront_structural_plane_stress`.
