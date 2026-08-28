# Forest--Rubin SRIX

**Mode:** explanation  
**Domain:** crystal-plasticity

SRIX is the current rate-independent FCC 316L crystal-plasticity path. It
resolves twelve slip systems, uses the FCC interaction matrix, isotropic
saturation and Armstrong--Frederick kinematic hardening. The point-local
state is driven by EBSD orientation and a three-dimensional elastic law.

The native implementation exists because a generic MFront 3-D bridge returns
an integrated response but does not expose the internal slip residual needed
for a coupled local plane-stress solve. MFront remains the qualified oracle;
the native path exposes an implementation architecture that can later move
from NumPy/Numba to CuPy/GPU.

Equations and Newton conventions are specified in
{doc}`../../reference/numerics/srix_semismooth_jacobian` and
{doc}`../../reference/numerics/native_srix_backend`.
