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

## Matrix capability contract

| Constitutive capability | CSR storage | PARDISO type | Default use |
|---|---|---:|---|
| `symmetric_positive_definite` | upper triangle | `mtype=2` | verified J2 behaviours |
| `nonsymmetric` | full matrix | `mtype=11` | unclassified laws and future crystal plasticity |

The relative tangent-asymmetry threshold is `1e-12`. A behaviour declaring
the SPD capability is rejected at runtime if its measured constitutive
tangent exceeds that threshold. The generic path does not silently symmetrize
a tangent.

## Fixed-pattern lifecycle

1. connectivity and constrained DOFs define the free-system pattern;
2. the fixed CSR assembler creates that pattern once;
3. phase 11 analyses it once;
4. each changed tangent updates only CSR numerical values;
5. phases 22 and 33 factorize and solve;
6. changing shape, `indptr` or `indices` after phase 11 is an error;
7. `close()` releases PARDISO memory and subsequent solves are rejected.

The matrix must be square, finite, `float64` CSR. Symmetric storage must not
contain entries below the diagonal. The right-hand side must be finite and
dimensionally compatible.

## Diagnostics

`FEMResult` records the matrix type, maximum tangent asymmetry, total and
per-phase PARDISO times, and analysis/factorization/solve call counts. For a
fixed-pattern nonlinear solve, `analysis_calls` must remain one, while
factorization and solve counts increase together. SciPy SuperLU is a
compatibility fallback; production campaigns require PyPardiso.
