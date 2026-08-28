# Reduced integration and hourglass control

**Mode:** explanation  
**Domain:** constitutive

Reduced integration can lower element cost, but the hourglass energy ratio is
not sufficient evidence that a plastic full-field solution is trustworthy.
The registered campaigns did not qualify CPS4R for the production plastic
workflow, so CPS4 remains the reference element for those comparisons.

The stabilised element uses

$$K_e=K^{1pt}(C_{tangent})+\beta(K^{4pt}_{ref}-K^{1pt}_{ref}),$$

with hourglass energy (E_{hg}=u_e^TK_{hg}u_e/2). At \(\beta=1\) this is
identical to CPS4 only in the linear elastic constant-tangent regime. After
yielding, the reference stabilisation remains elastic while the constitutive
tangent softens; the hourglass energy ratio is therefore not a proxy for
plastic-field error. The registered CPS4R campaign was not qualified even when
that ratio was small.

This is a negative qualification result, not a claim that reduced integration
is impossible in every future formulation.

The element, quadrature and energy contract is in
{doc}`../../reference/numerics/cps4r_hourglass`.
