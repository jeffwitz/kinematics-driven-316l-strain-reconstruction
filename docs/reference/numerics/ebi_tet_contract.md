# EBI-TET contract

EBI-TET has two kinematic samples and one constitutive state per pixel:

```text
sample_strain        (nx, ny, 2, 3)
mean_strain          (nx, ny, 3)
sample_stress        (nx, ny, 2, 3)
material states      nx * ny
```

One material evaluation is made at the mean strain. Sample stresses use the
elastic condensed tangent, never the algorithmic tangent. The algorithmic
tangent is used only for the mean-strain part of the matrix-free Jacobian.
Sample stresses are never averaged before the adjoint divergence.

EBI is an experimental adaptation for SRIX, not a generally valid contract
for arbitrary path-dependent materials.
