# Méric--Cailletaud reference

**Mode:** reference  
**Domain:** crystal-plasticity

The comparison branch is a twelve-system FCC single-crystal law with cubic
elasticity, pointwise orientation, isotropic and kinematic internal variables,
and a viscous slip evolution. Its state is advanced in the declared
pseudo-time increments; changing the increment partition changes the numerical
response and must be recorded.

| Aspect | SRIX | Méric--Cailletaud |
|---|---|---|
| FCC systems | 12 | 12 |
| anisotropy/orientation | yes | yes |
| physical viscosity | no in this use | yes |
| time-step partition dependence | not physical | material evolution depends on it |
| current role | production quasi-static path | comparison branch |

Parameter names, units, orientation convention and structural plane-stress
components are recorded in the selected behaviour manifest. Comparisons must
use identical orientations, systems, loading history and parameter provenance.
