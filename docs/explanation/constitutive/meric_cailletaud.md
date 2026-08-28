# Méric--Cailletaud

**Mode:** explanation  
**Domain:** constitutive

Méric--Cailletaud is retained as an important comparison branch. Its viscous
evolution introduces a dependence on the temporal discretisation that SRIX,
as used here, deliberately does not have. Comparisons must therefore use the
same orientations, slip systems, parameter provenance and loading history.

The recorded P43 comparison shows the practical consequence: the eight-step
run fails to converge while a refined sixteen-step path can converge.  This is
evidence of time-step sensitivity, not evidence that the fields have reached a
time-converged limit.  The comparison reports active-system overlap and
amplitude metrics separately; similar localisation does not imply identical
constitutive evolution.

The current production choice of SRIX does not erase the Méric results; it
states which model is used for the present rate-independent workflow and why.

The variables, units and evolution contract are given in
{doc}`../../reference/scientific/meric_cailletaud`; the reproducible comparison
procedure is in {doc}`../../how-to/reproduce/reproduce_srix_meric_comparison`.
