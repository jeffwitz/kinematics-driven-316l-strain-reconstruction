# P43 native SRIX plane-stress closure comparison

The native NumPy SRIX backend was run on the P43 M20 path with EBSD element
mapping in order F, the identified F parameters, tangent transverse predictor,
and the Numba LU12 local solver.  The constitutive backend and parameters are
identical; only the plane-stress driver changes.

| closure | wall time (s) | GMRES iterations | verification residual | RAW RMS (mm) |
|---|---:|---:|---:|---:|
| nested | 20.4185 | 2303 | 6.998e-09 | 3.952271743e-06 |
| coupled | 21.5790 | 2984 | 3.070e-12 | 3.952271743e-06 |

The coupled and nested displacement fields agree to a maximum absolute
difference of `1.94e-12 mm` (relative L2 difference `1.75e-12`).  The in-plane
stress fields differ by at most `1.32e-4 MPa` (relative L2 `1.44e-8`), and the
EVM fields differ by at most `5.58e-10` in the stored strain units.

The coupled closure therefore solves the same local problem to the tested
precision, but is not yet faster on this M20 case.  The nested implementation
remains the reference path and default; coupled is opt-in for the native NumPy
backend only.  MFront remains nested because its internal constitutive
residual/Jacobian is not exposed through the generic 3-D interface.

Artifacts:

* `reference_data/p0043_m20_numpy_srix_nested_v1/`
* `reference_data/p0043_m20_numpy_srix_coupled_v1/`

Both runs used 32 load increments and the same F-order EBSD mapping.  The
coupled path currently reports zero nested-material counters because it uses
the native coupled driver; wall time and global verification residual are the
authoritative diagnostics for this comparison.
