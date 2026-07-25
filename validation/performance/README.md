# Micromorphic hot-path optimization evidence

This directory preserves the paired measurements used to validate commit
`d5b0e7e` against baseline `decfe3d`.

- `nonlocal-hot-path-p0187/` contains the machine-readable constitutive
  reports for an intermediate real DIC crop.
- `nonlocal-hot-path-p0043/` contains the same paired benchmark on the complete
  P43 core selected for the next scientific campaign.
- `full-solver-p0187-*.log` contains the complete FEM gate, including solver
  diagnostics.
- `nonlocal_hot_path_optimization.json` is the consolidated comparison:
  parameters, hashes, timing ratios, memory ratios, field errors, and
  convergence-sequence identities.
- `fixed_csr_explicit_pardiso_p0187.json` is the second full-solver gate. It
  compares the optimized constitutive baseline with the fixed free-free CSR
  structure and explicit PARDISO phases 11/22/33 on the same P187 problem.

The large NumPy field snapshots remain under `results/performance/` and are
excluded from Git. The committed reports retain their hashes and numerical
comparisons without duplicating those binary arrays.

These results compare the same DIC zone, material maps, MFront thread count,
length, coupling modulus, tolerances, fixed point, Newton settings, increment
request, and nonsymmetric PARDISO matrix type. The first comparison changes
only the constitutive hot path. The second changes only sparse assembly and
the PARDISO phase cycle. They must not be used to attribute gains caused by
changing any scientific or numerical parameter.
