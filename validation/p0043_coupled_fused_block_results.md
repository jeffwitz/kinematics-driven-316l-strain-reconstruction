# P43 — fused coupled A/B–Schur kernel

An experimental Numba point-local kernel now fuses construction of the coupled
`A`/`B` blocks, one LU factorization of `A`, the four right-hand sides, Schur
formation, and the local `(delta_gamma, delta_eps_b)` correction.  The existing
NumPy/LAPACK path remains the oracle and is selected with
`coupled_block_solver="numpy"`.

## Qualification

* Unit tests: 16/16 pass.
* M20 fused vs oracle: displacement max difference `2.8e-17 mm`, stress max
  difference `2.8e-9 MPa`, EVM max difference `5.1e-15`.
* M100 fused vs direct-tangent oracle: displacement max difference
  `2.8e-17 mm`, stress max difference `4.0e-9 MPa`, EVM max difference
  `9.9e-15`.

## Timing

| case | wall time (s) | coupled block time (s) | global Newton | GMRES |
|---|---:|---:|---:|---:|
| M20, NumPy block | 17.905 | 6.188 | 146 | 2984 |
| M20, fused block | 14.507 | 3.261 | 146 | 2984 |
| M100, NumPy block | 324.269 | 109.429 | 140 | 3926 |
| M100, fused block | 224.977 | 51.940 | 140 | 3926 |

The M100 wall-time reduction is `30.6%` relative to the direct-tangent NumPy
block path.  The fused kernel changes no constitutive equations or global
solver settings; it only removes batch temporaries and keeps the local work in
one compiled point kernel.

Artifacts:

* `validation/reference_data/p0043_m20_numpy_srix_coupled_fused_block_v2/`
* `validation/reference_data/p0043_m100_numpy_srix_coupled_fused_block_v1/`
