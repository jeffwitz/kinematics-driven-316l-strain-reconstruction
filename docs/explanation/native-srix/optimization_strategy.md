# How the native CPU path was optimised

**Mode:** explanation  
**Domain:** architecture

The optimisation sequence preserved the equations at every step:

```text
naive NumPy
→ qualification against MFront
→ reduced 12-slip tangent and batch vectorisation
→ fixed-size LU12 kernels
→ coupled plane-stress closure
→ direct plane-stress tangent
→ fused A/B–Schur blocks
→ fused tangent and large-batch state kernels
→ adaptive dispatch by pending batch size
```

NumPy remains preferable for regular dense batch algebra; Numba is used for
irregular fixed-size point-local solves and fused blocks. Benchmarks are
machine-dependent and are evidence about implementation cost, not new
material parameters. The benchmark archive and exact commands are linked from
{doc}`../../how-to/crystal-plasticity/qualify_native_srix_backend`.

## Benchmark ledger

These representative runs are not additive speedup claims: wall time depends
on CPU load, warm-up, thread settings and the global Newton trajectory.

| P43 run | Wall time | Global Newton / GMRES | Interpretation |
|---|---:|---:|---|
| Native M20 nested | 20.418 s | 121 / 2303 | Native reference closure |
| Native M20 coupled | 21.579 s | 146 / 2984 | Same local problem, different trajectory |
| Native M100 nested | 477.091 s | 124 / 3390 | Scaling reference |
| Native M100 coupled | 379.395 s | 140 / 3926 | Lower local redundancy despite more global work |
| Native M100 fused block | 224.977 s | 140 / 3926 | Point-local A/B--LU12--Schur kernel |
| Native M100 fused-state | 211.687 s | 124 / 3390 | Indicative only; trajectory differs |

Direct MFront P43 references are also archived: M20 F mapping is `13.228 s`
with 2379 GMRES, while M100 F mapping is `358.237 s` elapsed (`326.244 s`
solver) with 7999 GMRES. They are the constitutive oracle, not a strict race
with the native scaling table because crop, history and trajectory provenance
differ. A fair timing comparison requires identical inputs and interleaved
warm-up runs.

The isolated benchmarks explain the hybrid design: fixed-size LU12 and fused
A/B--Schur remove local allocations and redundant solves, while the Dask
threaded prototype was not retained because its graph was rebuilt inside each
small constitutive call. Full reports remain under `validation/` and are
indexed by the evidence registry.
