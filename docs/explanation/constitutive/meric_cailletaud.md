# Méric--Cailletaud

**Mode:** explanation  
**Domain:** constitutive

Méric--Cailletaud is a small-strain, twelve-system FCC crystal-plasticity law
retained as an important comparison branch. For a Schmid tensor $M_s$, the
resolved shear is

$$
\tau_s=\boldsymbol{\sigma}:M_s.
$$

Its viscoplastic slip update is

$$
\Delta\gamma_s=\Delta t\left\langle
\frac{|\tau_s-X_s|-r_s}{K}\right\rangle^n
\operatorname{sign}(\tau_s-X_s).
$$

Here $X_s$ is the kinematic backstress, $r_s$ is the slip resistance, and $K$
and $n$ control the Norton sensitivity. The state retains signed slip
$\gamma_s$, accumulated slip $p_s$ (the increment is $|\Delta\gamma_s|$),
back-strain $a_s$, and therefore $X_s$ through the Armstrong--Frederick
parameters $C,d$. The isotropic resistance uses the FCC interaction matrix and
the saturation parameters $\tau_0,Q,b$.

The explicit $\Delta t$ is a physical loading-rate input of this formulation.
That is different from the numerical partition of a fixed physical path and
from local Newton robustness. A different physical rate would be a different
experiment; refining increments should instead converge for the same
prescribed history.

The recorded P43 comparison shows the practical consequence: the eight-step
run fails to converge while a refined sixteen-step path can converge. This is
evidence of numerical increment/solver sensitivity, not by itself evidence of
physical rate dependence or a time-converged field. The comparison reports
active-system overlap and amplitude metrics separately; similar localisation
does not imply identical constitutive evolution.

## Why this branch matters

Méric and SRIX share the FCC systems, orientation-dependent elasticity,
isotropic hardening, kinematic hardening and memory of the loading path. They
are not the same law: Méric uses a physical clock and rate-sensitive flow,
whereas SRIX uses an incremental rate-independent transition. Méric is thus a
controlled comparison and sensitivity branch, not a claim that SRIX is its
universal replacement.

The registered paired values are $K=12$ MPa, $n=11$, $\tau_0=40$ MPa,
$Q=10$ MPa, $b=3$, $C=40000$ MPa and $d=1500$; exact units, elastic constants
and preset provenance remain in the Reference page.

The primary formulation is the Méric--Cailletaud behaviour recorded in the
current Reference contract; the source and preset provenance are kept with the
registered MFront behaviour.

## Status boundary

* **Law/formulation:** the FCC viscoplastic flow and memory variables are
  defined above and in the registered MFront behaviour.
* **Implementation:** the P43 eight-step failure and sixteen-step diagnostic
  are qualification evidence for numerical robustness, not a convergence or
  material-rate claim.
* **Material calibration:** the paired values are a controlled literature
  preset; no experimental P43 Méric calibration is established.

The variables, units and evolution contract are given in
{doc}`../../reference/scientific/meric_cailletaud`; the reproducible comparison
procedure is in
{doc}`../../how-to/reproduce/reproduce_srix_meric_comparison`.
