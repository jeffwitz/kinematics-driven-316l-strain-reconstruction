# Forest--Rubin SRIX

**Mode:** explanation  
**Domain:** crystal-plasticity

SRIX is the current rate-independent FCC 316L crystal-plasticity path. It
resolves twelve slip systems, uses the FCC interaction matrix, isotropic
saturation and Armstrong--Frederick kinematic hardening. The point-local
state is driven by EBSD orientation and a three-dimensional elastic law.

For system (s), the inelastic increment is

$$\Delta\varepsilon^p=\sum_s\Delta\gamma_s M_s,$$

and the resistance is

$$r_s=\tau_0+Q\sum_jm_{sj}(1-e^{-bp_j}),\qquad
\Delta p_s=|\Delta\gamma_s|.$$

The signed slip drives the resolved shear while the accumulated absolute slip
drives isotropic hardening.  The kinematic backstress uses the
Armstrong--Frederick update in the qualified implementation.  The
rate-independent SRIX transition is controlled by ``R``, not by a physical
viscosity or by the duration of a pseudo-time step. In the qualified
incremental formulation the active-system update is

$$
\Delta\gamma_s = \Delta\varepsilon_{\rm eq}
\left\langle\frac{f_s}{R}\right\rangle
\operatorname{sign}(\tau_s-X_s),
$$

where ``Deq`` is assembled from the implicit strain and slip unknowns. Thus
``R`` is an overstress modulus, not a viscosity, and ``Deq`` is a strain
increment rather than a clock rate. Constructing ``Deq`` from the Newton
unknowns is part of the consistent tangent; replacing it with a fixed imposed
increment changes the derivative.

The currently registered ``R`` is an analytical transposition of a
Méric--Cailletaud ``(K,n)`` pair at a reference strain rate. It is explicitly
marked as ``analytical_transposition`` in the parameter provenance and is not
an identified 316L SRIX parameter.

The native implementation exists because a generic MFront 3-D bridge returns
an integrated response but does not expose the internal slip residual needed
for a coupled local plane-stress solve. MFront remains the qualified oracle;
the native path exposes an implementation architecture that can later move
from NumPy/Numba to CuPy/GPU.

The law is small-strain and local.  It does not by itself represent finite
lattice rotation, damage, viscosity or through-thickness heterogeneity, and a
registered parameter set is not automatically an identified 316L material.

Equations and Newton conventions are specified in
{doc}`../../reference/numerics/srix_semismooth_jacobian` and
{doc}`../../reference/numerics/native_srix_backend`.
