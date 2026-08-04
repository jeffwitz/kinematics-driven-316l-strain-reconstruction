# First full-Dirichlet spectral reconstruction

This tutorial follows the smallest reproducible path:

1. construct a rectangular nodal displacement field;
2. split it into harmonic applied and zero-boundary fluctuation parts;
3. verify the DST-I round trip;
4. evaluate the two TET2 strain samples;
5. apply the B0-preconditioned Newton-GMRES solve;
6. inspect the verified residual and reactions.

Begin with the existing qualification command:

```bash
python scripts/qualify_spectral2d_against_newton.py --help
```

Use an elastic material first. Only after the operator, transaction and
reaction checks pass should the registered SRIX history be enabled.
