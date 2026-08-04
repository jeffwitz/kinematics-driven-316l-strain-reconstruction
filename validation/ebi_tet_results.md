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

## State-sharing decomposition

The direct two-state TRI2 Newton-GMRES oracle uses the same kinematics, DST-I,
B0, line search, and global residual. At 8, 12, and 24 pixels per side, the
plastic slip error is:

| Grid | TET2 vs CPS4 | EBI vs TET2 | EBI vs CPS4 |
|---:|---:|---:|---:|
| 8 | 1.78% | 6.52% | 7.34% |
| 12 | 1.42% | 6.62% | 7.36% |
| 24 | 0.72% | 5.39% | 5.76% |

The dominant error is therefore the shared SRIX state, not the two-triangle
stencil. Side-resultant errors are smaller than nodal reaction errors; the
24x24 TET2-vs-CPS4 side-resultant error is `0.11%`, while its nodal reaction
error is `0.87%`.

Changing the actual B0 shape through `lambda_0/mu_0` ratios `{0.5, 1, 2}` gives
946, 967, and 975 GMRES iterations at 12x12. The earlier common scalar scaling
test was correctly uninformative and is no longer used as the B0 conclusion.

Raw JSON, NPZ fields, and SHA-256 field digests are under
`validation/_generated/ebi_tet/`.
