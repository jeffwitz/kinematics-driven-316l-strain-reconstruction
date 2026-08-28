# Structural plane stress is a three-traction reduction

**Mode:** explanation  
**Domain:** plane-stress

The three-dimensional law is used in a two-dimensional solver by relaxing
three transverse strains until

\[
\sigma_{zz}=\sigma_{xz}=\sigma_{yz}=0.
\]

This is not the same as imposing only \(\sigma_{zz}=0\). Crystal orientation
and Kelvin shear conventions affect all three equations. The nested closure
solves the 3-D material response inside an outer transverse Newton iteration;
the native coupled closure solves slips and transverse strains together. They
target the same local problem, while the generic MFront bridge naturally uses
the nested strategy.

The exact interface and Schur complement are in
{doc}`../../reference/numerics/plane_stress` and
{doc}`../../reference/numerics/native_srix_backend`.

The relaxed local variables are

$$\varepsilon_b=(\varepsilon_{zz},\varepsilon_{xz},\varepsilon_{yz}),$$

and the three residual constraints are all required:

$$\sigma_b=(\sigma_{zz},\sigma_{xz},\sigma_{yz})=0.$$

With (a=(xx,yy,xy)) and (b=(zz,xz,yz)), the condensed tangent is

$$C^{PS}=C_{aa}-C_{ab}C_{bb}^{-1}C_{ba}.$$

The nested, external-condensation, structural-MFront and native-coupled
implementations solve the same physical constraint; they differ in where the
local Newton work is performed.  The reduction is valid only for the declared
small-strain structural setting and its orientation/Kelvin conventions.
