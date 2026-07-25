# Three-backend FEM performance reference

This immutable campaign compares all three usable plane-stress material paths
on the same central 100×100-element crop of the versioned DIC case:

- `python`;
- `mfront-native-plane-stress`;
- `mfront-3d-condensed-plane-stress`.

Each backend is run three times in a fresh process. The execution order changes
between repetitions. All runs use 20 load increments, the analytical
J2/Ludwik law, two MKL threads, and two MGIS threads where applicable.
GNU `time` measures the peak resident set of the complete process. Every run
preserves its complete fields, solver diagnostics, stdout/stderr, and resource
record.

## Median results

| Backend | Process wall | FEM solver | Constitutive | Peak RSS | Global Newton |
|---|---:|---:|---:|---:|---:|
| Python J2 | 134.36 s | 133.31 s | 99.33 s | 248.96 MiB | 183 |
| MFront native plane stress | 27.03 s | 25.89 s | 9.52 s | 269.65 MiB | 93 |
| MFront 3D condensed | 83.43 s | 82.30 s | 65.44 s | 320.30 MiB | 93 |

All nine runs converge without cutback or local failure.

Relative to native MFront:

- Python takes `4.97×` more process wall time and `10.44×` more constitutive
  time, while using `7.7%` less peak RSS.
- The condensed 3D path takes `3.09×` more process wall time and `6.88×` more
  constitutive time, while using `18.8%` more peak RSS.
- The condensed path remains `1.61×` faster than Python in process wall time
  because it uses the MFront tangent and requires 93 rather than 183 global
  Newton iterations.

The process wall time includes input loading and compressed result
serialization. `FEM solver` is the internal `SolverDiagnostics.elapsed_seconds`
and excludes the final NPZ write.

## Numerical equivalence

Native MFront and condensed 3D MFront are equivalent to numerical precision:

| Field | Maximum absolute difference | Relative Linf |
|---|---:|---:|
| displacement | `9.171e-15 mm` | `1.282e-13` |
| stress | `2.307e-7 MPa` | `6.553e-10` |
| total strain | `3.197e-12` | `1.904e-10` |
| plastic strain | `3.267e-12` | `2.021e-10` |
| PEEQ | `2.427e-12` | `1.932e-10` |

Python is an independent implementation and is equivalent within the declared
case-study tolerances, not bitwise identical:

| Field | Maximum absolute difference | Relative Linf |
|---|---:|---:|
| displacement | `2.121e-9 mm` | `2.964e-8` |
| stress | `6.763e-2 MPa` | `1.921e-4` |
| total strain | `4.471e-7` | `2.661e-5` |
| plastic strain | `6.037e-7` | `3.734e-5` |
| PEEQ | `4.759e-7` | `3.788e-5` |

The maximum Gauss-point transverse residual is zero for Python,
`9.107e-14 MPa` for native MFront, and `3.745e-8 MPa` for condensed 3D
MFront. The condensed local Newton uses at most four iterations; its maximum
observed `cond(Cbb)` is `3.984`.

## Reproduction

```bash
source /home/jeff/.local/share/tfel/env/env.sh
export PYTHONPATH="/home/jeff/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export MFRONT_BEHAVIOUR_LIBRARY="$PWD/build/mfront/src/libBehaviour.so"
.venv/bin/python scripts/benchmark_fem_backends.py \
  --input data/processed/case_study \
  --output validation/reference_data/plane_stress_backend_performance_100x100_v1 \
  --library build/mfront/src/libBehaviour.so \
  --nx 100 \
  --ny 100 \
  --increments 20 \
  --repeats 3 \
  --mfront-threads 2 \
  --linear-threads 2
```

The command refuses to overwrite a non-empty campaign. Use another output
directory for a new measurement.
