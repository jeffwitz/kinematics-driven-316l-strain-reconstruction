# Native SRIX / Numba qualification evidence

**Mode:** reference  
**Domain:** evidence

The native path is qualified progressively:

```text
MFront 3-D nested -> native NumPy nested -> native coupled
                  -> Numba fused kernels
```

Each comparison records stress, state variables, condensed tangent,
plane-stress residual and final displacement before recording performance. The
primary M100 optimization reports are
`validation/p0043_srix_m100_coupled_optimization_results.md`,
`validation/p0043_m100_direct_tangent_profile.md` and
`validation/p0043_srix_m100_coupled_profile.md`; the M20 qualification reports
are indexed by the evidence registry. The performance ledger reports wall
time, local iterations, Newton/GMRES counts, batch size, threads and machine
conditions. A faster kernel is not by itself a correctness claim; field and
tangent comparisons establish equivalence.

The implementation contract is in
{doc}`../numerics/native_srix_backend`, and primary reports are indexed by
{doc}`evidence_registry`.
