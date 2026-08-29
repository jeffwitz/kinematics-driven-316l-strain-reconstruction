# Forest--Rubin SRIX

**Mode:** explanation  
**Domain:** crystal-plasticity

SRIX is the current rate-independent FCC 316L crystal-plasticity path. It
resolves twelve slip systems, uses the FCC interaction matrix, isotropic
saturation and Armstrong--Frederick kinematic hardening. The point-local
state is driven by EBSD orientation and a three-dimensional elastic law.

For system $s$, let $M_s$ be its symmetric Schmid tensor. The inelastic
increment and resolved shear stress are

$$
\Delta\varepsilon^p=\sum_s\Delta\gamma_s M_s,
\qquad
\tau_s=\boldsymbol\sigma:M_s.
$$

The committed state contains signed slip $\gamma_s$; $\Delta\gamma_s$ denotes
its increment. It also contains accumulated slip $p_s$, resistance $r_s$ and
kinematic backstress $X_s$. The Schmid tensors and cubic elastic constants are
rotated according to the local EBSD orientation.

The slip resistance is

$$
r_s=\tau_0+Q\sum_jm_{sj}(1-e^{-bp_j}),
\qquad \Delta p_s=|\Delta\gamma_s|.
$$

The signed slip drives the resolved shear while accumulated absolute slip
drives isotropic hardening. The kinematic backstress uses the
Armstrong--Frederick update in the qualified implementation.

The rate-independent SRIX transition is controlled by ``R``, not by a physical
viscosity or by the duration of a pseudo-time step. In the qualified
incremental formulation the active-system update is

$$
\Delta\gamma_s = \Delta\varepsilon_{eq}
\left\langle\frac{f_s}{R}\right\rangle
\operatorname{sign}(\tau_s-X_s).
$$

Here ``Deq`` is assembled from the implicit strain and slip unknowns. More
explicitly,

$$
\Delta\varepsilon_{eq}=\sqrt{\frac{2}{3}\,
\Delta e_{dev}:\Delta e_{dev}},
$$

with $\Delta e_{dev}$ formed from the elastic deviatoric increment and the
twelve slip contributions. At convergence this agrees with the value
reconstructed from the imposed total increment, but during Newton it does not
have the same derivative:

:::{admonition} Consistent-tangent point
The converged value may be identical while the Newton derivative is not.
Building $\Delta\varepsilon_{eq}$ from the unknowns keeps its dependence in
the Jacobian. Building it from the imposed ``deto`` treats it as constant and
misses the corresponding derivative term.
:::

The currently registered ``R`` is an analytical transposition of a
Méric--Cailletaud ``(K,n)`` pair at a reference strain rate. It is explicitly
marked as ``analytical_transposition`` in the parameter provenance and is not
an identified 316L SRIX parameter.

The native implementation exists because a generic MFront 3-D bridge returns
an integrated response but does not expose the internal slip residual needed
for a coupled local plane-stress solve. MFront remains the qualified oracle;
native SRIX exposes an implementation architecture that can later move from
NumPy/Numba to GPU.

The law is small-strain and local. It does not by itself represent finite
lattice rotation, damage, viscosity or through-thickness heterogeneity, and a
registered parameter set is not automatically an identified 316L material.

The primary Forest--Rubin formulation is the paper cited in the source
contract `mfront/Fcc316LForestRubinSrix.mfront`; the current Reference contract
records the associated FCC interaction and hardening provenance.

## Status boundary

* **Law/formulation:** Forest--Rubin SRIX is the registered rate-independent
  FCC formulation; its orientation and twelve-system state are defined above.
* **Implementation:** MFront is the qualified constitutive oracle; native SRIX
  is an independent implementation of the same registered law for coupled
  plane stress and acceleration.
* **Material calibration:** the current ``R`` is analytically transposed from
  a Méric ``(K,n)`` pair. A registered parameter preset is not an experimental
  316L identification.

Equations and Newton conventions are specified in
{doc}`../../reference/numerics/srix_semismooth_jacobian` and
{doc}`../../reference/numerics/native_srix_backend`.
