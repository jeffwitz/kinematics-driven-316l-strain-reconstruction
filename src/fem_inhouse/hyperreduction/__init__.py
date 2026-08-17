"""Constitutive hyper-reduction on a reduced integration domain.

The displacement space is **not** reduced and the equilibrium operator is
untouched. Only the nonlinear constitutive correction around a globally
evaluated elastic reference is sampled and reconstructed. See
`validation/constitutive_hyperreduction_preregistration.md`.
"""

from fem_inhouse.hyperreduction.split import (
    ConstitutiveSplit,
    ReferenceSplitTrial,
    reference_stiffness_of,
)

__all__ = ["ConstitutiveSplit", "ReferenceSplitTrial", "reference_stiffness_of"]
