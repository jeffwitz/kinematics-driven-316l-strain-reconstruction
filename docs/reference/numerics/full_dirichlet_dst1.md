# Full-Dirichlet DST-I contract

The fluctuation is zero on every boundary. Only interior nodes are transformed
with an orthonormal DST-I in both directions:

```python
dstn(field, type=1, norm="ortho")
idstn(transformed, type=1, norm="ortho")
```

The round trip must be below `1e-13` relative error. Boundary values remain
exactly zero after inverse reconstruction. Frequencies are
$\theta_x=\pi k/n_x$, $\theta_y=\pi l/n_y$, with
$k=1,\ldots,n_x-1$ and $l=1,\ldots,n_y-1$.

This page defines the mathematical transform contract only. The implementation
backend is selected separately through
{doc}`transform_backends`; SciPy is the default and FFTW is an optional
functionally equivalent backend. A numerical equivalence check does not imply
that a global FFTW performance advantage has been established. Mixed-boundary
transforms are outside this contract.
