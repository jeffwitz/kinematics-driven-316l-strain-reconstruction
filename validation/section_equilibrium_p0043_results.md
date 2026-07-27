# P43 generalized section-equilibrium baseline

Date: 2026-07-27

The preregistered diagnostic was executed on the archived local case and the
three coupled cases at `alpha = 1, 2, 4`. The machine-readable source is
`validation/reference_data/section_equilibrium_p0043_v1/report.json`.

## Result

| Case | Region | Section-force dispersion | Balance RMS / mean force | Closure gained from lateral flux |
|---|---|---:|---:|---:|
| local | padded solve domain | 5.447% | 2.272e-4 | 70.9% |
| local | retained core | 3.085% | 2.223e-4 | 69.6% |
| alpha 1 | padded solve domain | 4.737% | 2.203e-4 | 69.7% |
| alpha 1 | retained core | 2.776% | 1.940e-4 | 68.7% |
| alpha 2 | padded solve domain | 4.518% | 2.196e-4 | 69.2% |
| alpha 2 | retained core | 2.652% | 1.739e-4 | 69.0% |
| alpha 4 | padded solve domain | 4.315% | 2.230e-4 | 68.5% |
| alpha 4 | retained core | 2.543% | 1.530e-4 | 69.3% |

The section force varies by several percent because P43 is an interior
Dirichlet partition. That variation is not an equilibrium error by itself.
Including the shear flux across the artificial lateral cuts removes about
69--71% of the naive adjacent-section imbalance. The remaining RMS residual
is 1.5e-4 to 2.3e-4 of the mean section force.

## Interpretation

This run establishes a reproducible numerical baseline, not a pass/fail
validation. The residual is evaluated from saved element-centred stresses,
and lateral tractions are approximated with the first and last cell centres.
It is therefore less exact than the quadrature-level finite-element residual.

The absolute mean resultants increase with coupling, but they must not be
compared with an experimental force: the available archive contains neither a
synchronised load-cell history nor a confirmed full gauge width. The 2 mm
thickness used here is the value reported by the article, not a traceable
measurement on the archived specimen.

## Reproduction

```bash
.venv/bin/fem-inhouse diagnose-section-equilibrium \
  --campaign local results/constitutive-local-p0043-pad150 \
  --campaign alpha_1 results/constitutive-nonlocal-p0043-pad150-a100 \
  --campaign alpha_2 results/constitutive-nonlocal-p0043-pad150-a200 \
  --campaign alpha_4 results/constitutive-nonlocal-p0043-pad150-a400 \
  --partition-id 43 \
  --thickness-mm 2.0 \
  --output validation/reference_data/section_equilibrium_p0043_v1
```
