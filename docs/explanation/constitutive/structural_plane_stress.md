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
