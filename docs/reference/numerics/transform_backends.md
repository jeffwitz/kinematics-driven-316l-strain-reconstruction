# Spectral transform backends

## Configuration

`SpectralTransformConfig` is shared by the full-Dirichlet spectral solvers:

| Field | Meaning | Default |
| --- | --- | --- |
| `backend` | `scipy` or `fftw` | `scipy` |
| `workers` | transform threads/workers | `1` |
| `fftw_planner_effort` | `estimate`, `measure`, or `patient` | `measure` |
| `fftw_planning_time_limit_s` | diagnostic planning budget | `2.0` |
| `fftw_wisdom_directory` | optional machine-local directory | `None` |
| `fftw_use_wisdom` | enable compatible wisdom | `True` |

Requesting `fftw` without pyFFTW raises an explicit `ImportError`; there is no
silent fallback to SciPy.

## Diagnostics contract

Every plan exposes `TransformDiagnostics` with:

```text
backend, implementation, interior_shape, batch_components, dtype,
workers, planner_effort, wisdom_loaded, planning_seconds
```

The solver copies these values into `Spectral2DDiagnostics`. The shape is
`(nx - 1, ny - 1)` and `batch_components` is `2`.

## Numerical contract

For a random interior displacement `u`, both backends must satisfy

```{math}
\frac{\lVert Q^{-1}Qu-u\rVert_2}{\lVert u\rVert_2} < 10^{-13}.
```

They must also preserve the Euclidean inner product and agree on the complete
`DST-I -> B0 -> inverse DST-I` preconditioner to the registered numerical
tolerance. These are functional contracts, not performance claims.
