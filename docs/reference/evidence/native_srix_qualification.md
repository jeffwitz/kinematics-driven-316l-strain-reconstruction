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
performance ledger reports wall time, local iterations, Newton/GMRES counts,
batch size, threads and machine conditions. A faster kernel is not by itself
a correctness claim; field and tangent comparisons establish equivalence.

The implementation contract is in
{doc}`../numerics/native_srix_backend`, and primary reports are indexed by
{doc}`evidence_registry`.
