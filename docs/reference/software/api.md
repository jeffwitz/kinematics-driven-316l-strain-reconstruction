# Python API reference

**Mode:** reference  
**Domain:** software

The public Python API is organised around case preparation, kinematics,
matrix-free global solves and material adapters.  Constitutive adapters expose
the transactional `evaluate`, `complete_trial`, `commit` and `revert`
contract; solver-facing plane-stress responses provide in-plane stress and
tangent.

Concrete signatures are generated from the maintained modules and tests.  A
caller must not mutate committed material state during an evaluation.
