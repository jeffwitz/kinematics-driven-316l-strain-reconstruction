# Sparse-solver reference

**Category: Reference.**

The mechanical sparsity graph is fixed by mesh connectivity and constrained
degrees of freedom. The solver builds the free-system CSR graph once and
updates only its numerical data during Newton.

PARDISO phase 11 performs symbolic analysis once. Phases 22 and 33 perform
numerical factorization and solve for each new tangent.

Verified J2 behaviours declare a symmetric positive-definite tangent
capability. Their free matrix stores the upper CSR triangle and uses
`mtype=2`. Unclassified or potentially nonsymmetric behaviours retain full CSR
storage and `mtype=11`. A future crystal-plasticity law therefore remains
nonsymmetric until its tangent contract has been verified.

Runtime checks reject a tangent whose measured asymmetry exceeds the declared
capability threshold. Timers separate constitutive integration, Kelvin
conversion, internal forces, element matrices, sparse assembly, free-system
extraction and PARDISO phases.
