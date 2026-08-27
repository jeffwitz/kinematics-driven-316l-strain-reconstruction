# P43 — fused direct coupled tangent

The direct coupled plane-stress tangent is now evaluated by a point-local
Numba kernel.  It constructs the reduced `A`, `E`, and `B` blocks locally,
solves the six right-hand sides, forms the 3x3 Schur complement, and returns
only the plane-stress tangent and predictor blocks.  The NumPy direct tangent
remains the oracle.

## Qualification

* Unit tests: 16/16 pass.
* M20 fused tangent vs the previous fused-block path: displacement max
  difference `2.8e-17 mm`, stress max difference `2.5e-9 MPa`, EVM max
  difference `5.4e-15`.
* M100 fused tangent vs fused-block path: displacement max difference
  `2.8e-17 mm`, stress max difference `4.0e-9 MPa`, EVM max difference
  `7.3e-15`.

## Timing

| case | wall time (s) | coupled block (s) | direct tangent (s) |
|---|---:|---:|---:|
| M20, fused block | 14.507 | 3.261 | 3.327 |
| M20, fused tangent | 12.924 | 3.107 | 2.310 |
| M100, fused block | 224.977 | 51.940 | 43.483 |
| M100, fused tangent | 243.721 | 59.769 | 33.373 |

The M20 wall-time gain is 10.9%.  On the single M100 runs, the tangent
sub-counter decreases by 23.2% (43.48 to 33.37 s), but wall time is higher by
8.3% because state, block, and line-search counters are also slower in that
run.  This is machine variability, not a field or trajectory change; a
paired/interleaved benchmark is required before claiming a M100 wall-time
gain.

The parallel Numba variant of this tangent kernel was not retained: its first
trial was unstable.  The qualified fused tangent path is serial point-local
Numba with `NUMBA_NUM_THREADS=1`; the constitutive equations and tolerances
are unchanged.

Artifacts:

* `validation/reference_data/p0043_m20_numpy_srix_coupled_fused_tangent_v3/`
* `validation/reference_data/p0043_m100_numpy_srix_coupled_fused_tangent_v1/`
