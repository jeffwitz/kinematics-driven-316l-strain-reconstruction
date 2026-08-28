# Native SRIX / Numba qualification evidence

**Mode:** reference  
**Domain:** evidence

The native path is qualified progressively. The reports below are the primary
evidence; they are not interchangeable with a general claim about every mesh
or machine:

```text
MFront 3-D nested -> native NumPy nested -> native coupled
                  -> Numba fused kernels
```

| Stage/claim | Primary artifact | Recorded result and boundary |
|---|---|---|
| coupled local formulation | `validation/p0043_direct_coupled_tangent_results.md` | direct implicit tangent agrees with the 3-D-condensed oracle below `2e-16` on the checked local/two-step states; no claim beyond those states |
| fused A/B--Schur kernel | `validation/p0043_coupled_fused_block_results.md` | M20 and M100 fields agree within `2.8e-17 mm` displacement and `1e-14` EVM; M100 block time `109.429→51.940 s` with `140/3926` Newton/GMRES unchanged |
| fused direct tangent | `validation/p0043_coupled_fused_tangent_results.md` | M20 tangent time `3.327→2.310 s`; M100 tangent counter `43.483→33.373 s`; the M100 wall-time comparison is machine-variable |
| adaptive fused-state dispatch | `validation/p0043_coupled_state_crossover_results.md` | ratios `1.43, 1.37, 1.38` at 6k--10k and `0.65, 0.94, 0.92` at 12k--20k in one sweep; default threshold `12000`, explicitly machine-dependent |

Each comparison records stress, state variables, condensed tangent,
plane-stress residual and final displacement before recording performance. The
primary M100 optimization reports are
`validation/p0043_srix_m100_coupled_optimization_results.md`,
`validation/p0043_m100_direct_tangent_profile.md` and
`validation/p0043_srix_m100_coupled_profile.md`; the M20 qualification reports
are indexed by the evidence registry. The performance ledger reports wall
time, local iterations, Newton/GMRES counts, batch size, threads and machine
conditions. A faster kernel is not by itself a correctness claim; field and
tangent comparisons establish equivalence. The native implementation is a
validated high-performance path for the registered configurations, not a claim
that SRIX parameters have been identified or that a GPU implementation already
exists.

The implementation contract is in
{doc}`../numerics/native_srix_backend`, and primary reports are indexed by
{doc}`evidence_registry`.
