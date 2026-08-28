# Méric--Cailletaud reference

**Mode:** reference  
**Domain:** crystal-plasticity

The comparison branch is a twelve-system FCC single-crystal law with cubic
elasticity, pointwise orientation, isotropic and kinematic internal variables,
and a viscous slip evolution. For each system, with resolved shear stress
``tau_s``, backstress ``X_s`` and resistance ``r_s``, the implicit update is
proportional to

$$
\Delta\gamma_s = \Delta t\,\left\langle\frac{|\tau_s-X_s|-r_s}{K}\right\rangle^n
\operatorname{sign}(\tau_s-X_s).
$$

The hardening parameters are the registered ``tau0, Q, b, C, d``; ``p_s`` is
the accumulated slip and ``a_s`` is the Armstrong--Frederick back-strain.
The complete registered parameter set has cubic ``C11=197000``,
``C12=125000`` and ``C44=122000`` MPa, ``n=11``, ``K=12`` MPa,
``tau0=40`` MPa, ``Q=10`` MPa, ``b=3``, ``C=40000`` MPa and ``d=1500``.
These values are provenance data, not a new material identification.

The explicit ``Delta t`` makes the law physically rate-dependent. Separately,
the number and placement of increments affects numerical integration and local
Newton robustness. The registered 8-increment failure and 16-increment
success therefore demonstrate a timestep/solver sensitivity of that campaign;
they do not, by themselves, establish a physical rate law or convergence of
the time-discrete fields.

| Aspect | SRIX | Méric--Cailletaud |
|---|---|---|
| FCC systems | 12 | 12 |
| anisotropy/orientation | yes | yes |
| physical loading-rate dependence | no in this use | yes |
| numerical increment partition | path-independent in this use | should converge when refined |
| observed current robustness | good | requires finer partition |
| current role | production quasi-static path | comparison branch |

Parameter names, units, orientation convention and structural plane-stress
components are recorded in the selected behaviour manifest. Comparisons must
use identical orientations, systems, loading history and parameter provenance.
