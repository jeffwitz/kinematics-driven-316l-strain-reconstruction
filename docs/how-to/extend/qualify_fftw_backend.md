# Qualify the FFTW backend

**Mode:** how-to  
**Domain:** spectral

Run the same transform and preconditioner workload with the SciPy and FFTW
backends after warming plans and buffers.  Compare numerical outputs,
residuals, iteration counts and wall time; store FFTW version, planner mode,
wisdom and thread count in the benchmark manifest.

This qualification concerns the transform backend only.  Do not attribute a
global solver speedup to FFTW without separating GMRES, material and assembly
timings.  See {doc}`../../reference/numerics/transform_backends`.
