# Optional FFTW backend

The spectral solver keeps SciPy as its reference transform backend and exposes
an optional explicit pyFFTW backend for machine-specific performance studies.
The backend changes only the implementation of the orthonormal DST-I. It does
not change the discrete symbols, the B0 preconditioner, the constitutive law,
or the Newton-GMRES algorithm.

:::{admonition} Repository adaptation
:class: note

The implementation uses `pyfftw.FFTW` with `FFTW_RODFT00`, two persistent
aligned buffers for each direction, and a final axis containing the two
displacement components. No padding or alternative frequency grid is used.
:::

## Orthonormal scaling

For interior dimensions `nx - 1` and `ny - 1`, an unnormalised two-dimensional
DST-I satisfies

```{math}
F^2 = 4 n_x n_y I.
```

Both the forward and inverse FFTW calls therefore apply

```{math}
Q = \frac{1}{2\sqrt{n_x n_y}}F.
```

The SciPy implementation uses `norm="ortho"`; the numerical contract requires
the two backends to agree without changing any modal symbol.

## Plans, buffers and wisdom

Plans are constructed once for a solver instance. The hot preconditioner path
uses `forward_into`, `B0Green2D.apply_into`, and `inverse_into` on persistent
arrays. The plan is never cached globally and is not shared between concurrent
solvers.

FFTW wisdom is optional and machine-local. Its metadata includes the array
shape, dtype, transformed axes, thread count, planner effort, Python and
library versions, and platform identity. Incompatible or corrupt wisdom is
ignored and causes a new plan to be built.

Planning time is reported separately from steady-state transform time. SciPy
remains the default until the recorded full-solver benchmark meets the
pre-registered performance criteria.
