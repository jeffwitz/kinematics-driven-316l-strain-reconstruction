# P43 SRIX elastic–plastic manifold probe (M20)

## Scope and environment

This is a registered-case methodological diagnostic. It uses the archived M20
SRIX displacement contract: scored steps `[4, 8, 12, 16, 20, 24, 28, 32]`,
eight `21 x 21 x 2` displacement blocks, 7056 rows, and millimetres.

The historical runtime was recovered on the host. The repaired history is
`validation/reference_data/dic_multistep_history_p0043_repaired_v1/repaired_history_mm.npy`
with SHA256
`8b0c6df9b8ac6235c87b0e5d60e5dee6a4e6e905980c879d50c595fe1d72c8a0`. MGIS was
loaded from the historical virtual environment with TFEL/MFront 5.1.0 and
`build/mfront/src/libBehaviour.so`. The baseline 32-step replay reproduces the
archived displacement exactly (`max_abs_delta = 0`, `relative_delta = 0`).

## Tangent anatomy

Elasticity was parameterised by stable cubic coordinates

\[
K=(C_{11}+2C_{12})/3,\qquad C'=(C_{11}-C_{12})/2,\qquad C_{44},
\]

in logarithmic form. The historical constants are `(197000, 125000, 122000)`
MPa. Six central elastic finite-difference forwards (`±1%`) and two half-step
checks were run. The half-step `K` derivative difference is `4.06e-4`.

The six retained combined directions use a relative SVD threshold of `1e-4`.
The seventh direction is a near-null dominated by `Q-b`:

| mode | σ/σ₁ | elastic weight | plastic weight | class |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1.0000 | 0.110 | 0.890 | plastic-dominated |
| 2 | 0.8167 | 0.909 | 0.091 | elastic-dominated |
| 3 | 0.2440 | 0.943 | 0.057 | elastic-dominated |
| 4 | 0.1307 | 0.912 | 0.088 | elastic-dominated |
| 5 | 0.0750 | 0.126 | 0.874 | plastic-dominated |
| 6 | 0.0156 | 0.0002 | 1.000 | plastic-dominated |

The near-null has normalized singular value `1.44e-5`, elastic weight
`5.8e-10`, and `(log Q, log b)` components approximately `(0.706, -0.709)`.

The leading mode is mixed but mostly plastic (`f_P=0.890`); modes 2–4 are
predominantly elastic. Thus the new elastic directions are not merely a small
rotation of the plastic hardening modes.

## Krylov and residual projections

The mode contributions below are squared coefficients in the six-dimensional
combined left-singular basis; each column sums to the corresponding combined
projection fraction.

| combined mode | Krylov raw | Krylov dissipative | final SRIX residual |
| ---: | ---: | ---: | ---: |
| 1 | 0.02975 | 0.03867 | 0.01234 |
| 2 | 0.02554 | 0.03220 | 0.28121 |
| 3 | 0.03596 | 0.01881 | 0.05545 |
| 4 | 0.09919 | 0.09919 | 0.05794 |
| 5 | 0.01045 | 0.00562 | 0.00021 |
| 6 | 0.01601 | 0.01090 | 0.00599 |
| **sum** | **0.21691** | **0.20539** | **0.41314** |

The accessible correction is therefore distributed over several modes rather
than carried by a single constitutive direction. The state-wise combined
fractions of the raw Krylov contribution (steps 4–32) are
`0.2105, 0.2926, 0.3271, 0.3287, 0.1942, 0.3891, 0.3950, 0.2778`; no simple
elastic-first/plastic-late separation appears.

## Updated-elasticity control

One additional forward used the recorded alternative constants
`(C11,C12,C44)=(218300,144800,125400)` MPa with baseline plastic parameters.
Its linearisation error is `0.0433`, and its residual ratio is `rho=1.0017`.
It is therefore a consistency control, not a calibration, and does not improve
the baseline residual. Its displacement is anti-aligned with the raw Krylov
correction (`c_K=-0.407`).

## Nonlinear combined probes

The first three modes passed the interpretability gate, so exactly six probes
were run at the ±20% maximum-component amplitude. All cubic stability checks
passed.

| mode/sign | ρ residual | linearisation error | χ⊥ | c_K,⊥ |
| --- | ---: | ---: | ---: | ---: |
| 1/+ | 0.9933 | 0.3590 | 0.1795 | 0.1541 |
| 1/− | 1.0093 | 0.3981 | 0.2575 | -0.0652 |
| 2/+ | 1.0185 | 0.1335 | 0.0976 | 0.0322 |
| 2/− | 0.9826 | 0.1062 | 0.0951 | -0.0903 |
| 3/+ | 1.0023 | 0.2388 | 0.1986 | 0.0071 |
| 3/− | 0.9979 | 0.2221 | 0.1808 | -0.0523 |

The best residual reduction is only about 1.7% (mode 2/−), and its transverse
component is anti-aligned with the Krylov transverse correction. The sole
positive transverse alignment of note (mode 1/+) is small and does not produce
a robust residual reduction.

## Verdict

**B — elasticity helps linearly, curvature does not rescue the missing
correction.** Elasticity adds substantial independent observable directions and
raises the local explained fraction from `0.0640` (plastic only) to `0.2169`
for raw Krylov, and from `0.2100` to `0.4131` for the final SRIX residual.
However, the ±20% nonlinear probes do not bend the manifold consistently toward
the 78% Krylov-transverse component. The next step should therefore not be a
seven-parameter FEMU. It should decide whether a targeted constitutive
structural extension can add the missing field directions.

All results remain limited to the P43 registered-case diagnostic; physical
DIC–EBSD co-registration is not independently proven (R2 is not satisfied).
Parameter factors in the JSON are tangent-equivalent conditioning diagnostics,
not identified material parameters. The generated sensitivity arrays remain
local/ignored; the committed script, JSON and report record their hash and
provenance without exporting the binary payload.
