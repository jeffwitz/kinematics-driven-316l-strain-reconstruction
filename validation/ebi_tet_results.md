# EBI-TET SRIX validation results

Status: `experimental_negative`.

The mandatory Hookean prerequisite passes for three orientations at initial,
elastic, and plastified states. The maximum full-3D error is `6.70e-16`; the
maximum condensed plane-stress error is `3.79e-14`.

The plastic EBI Jacobian directional test passes with a minimum relative error of
`8.58e-7`. The elastic EBI reconstruction is exactly identical to traditional
two-triangle integration, while using one material state and one material batch
evaluation per pixel.

On the registered homogeneous SRIX case, Newton-GMRES reaches `1e-8` without
stabilisation. At 12x12 its verified residual is `1.12e-12`, with 32 Newton
iterations, 967 GMRES iterations, and 64 material evaluations. Changing the B0
scale over `{0.5, 1, 2}` leaves the converged fields and deterministic counters
unchanged.

| Grid | Eu | Esigma | EGamma | ER |
|---:|---:|---:|---:|---:|
| 4 | 2.28% | 2.29% | 4.82% | 31.96% |
| 8 | 2.10% | 2.26% | 7.34% | 14.48% |
| 12 | 1.76% | 1.90% | 7.36% | 10.06% |
| 24 | 1.11% | 1.23% | 5.76% | 6.01% |

The 48x48 CPS4 oracle failed by constitutive cutback, so it cannot extend the
comparison. The one-point witness failed to reach `1e-6` after 7,928 iterations;
its final high-frequency fraction was `1.43e-4`, versus a maximum `1.04e-7` for
EBI at 12x12.

Raw JSON, NPZ fields, and SHA-256 field digests are under
`validation/_generated/ebi_tet/`.

