# Méric--Cailletaud

**Mode:** explanation  
**Domain:** constitutive

Méric--Cailletaud is retained as an important comparison branch. Its viscous
evolution introduces physical loading-rate dependence that SRIX, as used here,
deliberately does not have. That physical dependence must be distinguished from
the numerical partition of a fixed loading history and from local Newton
robustness. Comparisons must therefore use the same orientations, slip systems,
parameter provenance and loading history.

The recorded P43 comparison shows the practical consequence: the eight-step
run fails to converge while a refined sixteen-step path can converge. This is
evidence of numerical increment/solver sensitivity, not by itself evidence of
physical rate dependence or a time-converged field. The comparison reports active-system overlap and
amplitude metrics separately; similar localisation does not imply identical
constitutive evolution.

The current production choice of SRIX does not erase the Méric results; it
states which model is used for the present rate-independent workflow and why.

The variables, units and evolution contract are given in
{doc}`../../reference/scientific/meric_cailletaud`; the reproducible comparison
procedure is in {doc}`../../how-to/reproduce/reproduce_srix_meric_comparison`.
