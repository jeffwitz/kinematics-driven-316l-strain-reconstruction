# Comparing SRIX and Méric--Cailletaud

The two laws share the same twelve FCC systems, orientation-dependent cubic
elasticity, interaction matrix, isotropic saturation and Armstrong--Frederick
kinematic hardening. They differ in the flow rule:

| aspect | Méric--Cailletaud | Forest--Rubin SRIX |
|---|---|---|
| physical rate dependence | yes, through $\Delta t$, $K$, $n$ | no in the registered use |
| loading-path dependence | yes | yes |
| main flow scale | Norton drag $K$ and exponent $n$ | overstress scale $R$ and $\Delta\varepsilon_{eq}$ |
| current role | controlled comparison branch | production quasi-static law |

The comparison is made at the level of signed slips, active-system sets and
spatial fields. Jaccard, cosine, Spearman and common-activity measures can
show similar localisation patterns without implying equal amplitudes or
identical constitutive histories. A refined numerical partition is not a
change of physical rate, and an eight-step Méric failure is not by itself a
viscosity measurement. The evidence page must therefore report both
similarities and differences.

MFront provides the qualified oracle for both behaviours; native SRIX is an
independent implementation of the SRIX law, not a third constitutive model.
The paired preset is a controlled formulation comparison, not two identified
material parameter sets.
