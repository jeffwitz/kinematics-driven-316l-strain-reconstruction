# MFront 3D condensation: DIC 10×10 reference

This immutable campaign compares two integrations of the same J2/Ludwik
behaviour on the versioned DIC-driven 10×10 case:

- native MFront `PlaneStress`;
- MFront `Tridimensional`, followed by a local solve for
  `[epsilon33, gamma13, gamma23]` and an explicit Schur-complement tangent.

The mesh, displacement unknowns, global Newton solve, and historical 2D result
contract are identical in both runs.

## Reproduction

From the repository root:

```bash
source /home/jeff/.local/share/tfel/env/env.sh
export PYTHONPATH="/home/jeff/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export MFRONT_BEHAVIOUR_LIBRARY="$PWD/build/mfront/src/libBehaviour.so"
.venv/bin/python scripts/compare_fem_backends.py \
  --input data/processed/case-study-10x10 \
  --output validation/reference_data/mfront_3d_condensed_dic_10x10_v1 \
  --library build/mfront/src/libBehaviour.so \
  --threads 2 \
  --reference-backend mfront-native-plane-stress \
  --prediction-backend mfront-3d-condensed-plane-stress
```

The script refuses to overwrite a non-empty campaign. Use another output
directory for a fresh run.

## Result

All field-comparison and invariant checks pass. Both backends converge in
20 increments and 66 global Newton iterations, without cutback.

| Quantity | Native plane stress | Condensed 3D |
|---|---:|---:|
| elapsed time (s) | 1.310 | 2.805 |
| constitutive time (s) | 0.377 | 1.804 |
| maximum Gauss-point transverse residual (MPa) | 5.575e-14 | 2.705e-08 |
| maximum local iterations | 0 | 4 |
| mean local iterations | 0 | 2.666 |
| maximum `cond(Cbb)` | 0 | 1.896 |
| local failures | 0 | 0 |

The maximum field differences are:

| Field | Maximum absolute difference | Relative Linf |
|---|---:|---:|
| displacement (mm) | 6.245e-16 | 8.743e-15 |
| in-plane stress (MPa) | 4.804e-08 | 2.069e-10 |
| total 3D strain | 5.101e-13 | 3.542e-10 |
| plastic 3D strain | 6.117e-13 | 6.783e-10 |
| PEEQ | 4.038e-13 | 3.015e-10 |

For the condensed backend, the element-output consistency checks give:

- maximum averaged transverse residual: `1.419e-08 MPa`;
- maximum `abs(trace(PE))`: `1.220e-19`;
- maximum `abs(E - EE - PE)`: `2.168e-19`.

`report.json` records the exact Git commit, input and library hashes,
configuration, thresholds, Gauss-point diagnostics, and artifact hashes.
