# Output-data contract

**Mode:** reference  
**Domain:** data

Public results contain nodal displacement, element stress/strain/plastic
fields, reactions, optional completed three-dimensional tensors and the
plane-stress residual.  Two-dimensional vectors use the declared engineering
shear convention; completed tensors use tensorial shear and the repository
Kelvin order.  Optional frames do not replace the final state.

Every result records backend identity, constitutive and plane-stress options,
convergence diagnostics, input hashes and campaign provenance.  See
{doc}`../scientific/tensor_conventions` and
{doc}`../numerics/plane_stress` for component and residual definitions.
