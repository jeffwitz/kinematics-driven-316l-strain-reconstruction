# Structural plane stress is a three-traction reduction

**Mode:** explanation  
**Domain:** plane-stress

The three-dimensional constitutive law is used in a two-dimensional structural
solver by relaxing three transverse strains until

$$
\sigma_{zz}=\sigma_{xz}=\sigma_{yz}=0.
$$

This is not the same as imposing only $\sigma_{zz}=0$. Crystal orientation and
Kelvin shear conventions affect all three equations. The systems of slip,
stress and internal variables remain three-dimensional; only the local closure
is reduced.

The relaxed local variables are

$$
\varepsilon_b=(\varepsilon_{zz},\varepsilon_{xz},\varepsilon_{yz}),
$$

and the three residual constraints are

$$
\sigma_b=(\sigma_{zz},\sigma_{xz},\sigma_{yz})=0.
$$

The nested closure solves the 3-D material response inside an outer transverse
Newton iteration; the native coupled closure solves slips and transverse
strains together. External condensation and the structural-MFront wrapper
provide other implementations of the same local physical constraint.

With $a=(xx,yy,xy)$ and $b=(zz,xz,yz)$, the condensed tangent is

$$
C^{PS}=C_{aa}-C_{ab}C_{bb}^{-1}C_{ba}.
$$

Standard two-dimensional MFront ``PlaneStress`` is not automatically
equivalent to this arbitrary-orientation crystal closure: the latter requires
all three transverse tractions to vanish. The reduction is valid only for the
declared small-strain structural setting and its orientation/Kelvin
conventions.

## Status boundary

* **Law/formulation:** a three-dimensional crystal law is closed by
  $\sigma_{zz}=\sigma_{xz}=\sigma_{yz}=0$.
* **Implementation:** nested, external condensation, structural-MFront and
  native-coupled routes differ in where the local Newton work is performed.
* **Material calibration:** plane-stress equivalence is a numerical/
  constitutive contract; it does not establish calibration of the underlying
  crystal law.

The exact interface and Schur complement are in
{doc}`../../reference/numerics/plane_stress` and
{doc}`../../reference/numerics/native_srix_backend`.
