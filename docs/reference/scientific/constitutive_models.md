# Constitutive-model contract

**Mode:** reference  
**Domain:** constitutive

The repository exposes three families of constitutive response:

- isotropic J2/Ludwik for baselines and diagnostics;
- Forest--Rubin SRIX for rate-independent FCC crystal plasticity;
- Méric--Cailletaud for the historical viscous crystal comparison.

The J2 baseline uses (f=\sigma_{eq}-R(p)) with regularised Ludwik hardening.
SRIX uses twelve FCC systems, cubic elasticity, an interaction matrix and
isotropic/kinematic hardening. Méric retains its time-dependent evolution.
Parameter provenance is defined in {doc}`../srix_parameter_sets`; the numerical
state and response contract is defined by the selected backend.
