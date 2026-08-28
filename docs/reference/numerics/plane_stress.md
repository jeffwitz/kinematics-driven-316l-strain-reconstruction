# Plane-stress numerical contract

**Mode:** reference  
**Domain:** plane-stress

The global finite-element problem has two displacement degrees of freedom per
node. Its strain vector is
$[\varepsilon_{11},\varepsilon_{22},\gamma_{12}]$ and its stress vector is
$[\sigma_{11},\sigma_{22},\sigma_{12}]$.

The three-traction crystal contract is not synonymous with every object named
``PlaneStress``. A standard two-dimensional MFront behaviour enforces its own
model-specific axial constraint. An anisotropic three-dimensional law used in
this repository instead relaxes three transverse strain components.

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

The available strategies are deliberately distinct:

| strategy | local state | zero tractions | local solve |
|---|---|---|---|
| standard MFront `PlaneStress` | behaviour-specific 2-D state | behaviour-defined axial constraint | inside the behaviour |
| external 3-D condensation | generic 3-D integrated state | `zz`, `xz`, `yz` | outer nested solve |
| `StructuralPlaneStress3D` | 3-D MFront state | `zz`, `xz`, `yz` | structural MFront wrapper |
| native SRIX nested | slips plus transverse strains | `zz`, `xz`, `yz` | nested local Newton |
| native SRIX coupled | slips plus transverse strains | `zz`, `xz`, `yz` | one block local Newton |

Only the last four entries implement the crystal three-traction contract. They
share the physical constraint but differ in where the local Newton solve and
consistent tangent are constructed; the generic MFront bridge naturally uses
the nested strategy.

The full structural derivation is in {doc}`mfront_structural_plane_stress`.
