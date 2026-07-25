# Article-sized MFront partition 48

This immutable campaign preserves the pre-registered partition used to select
the scalar Helmholtz diagnostic length.

## Scope

| Property | Value |
|---|---:|
| global ROI | `3600×3100` elements |
| layout | `10×10`, 100 partitions |
| partition | `48`, index `(4,8)` |
| retained core | `360×310` elements |
| solved region | `660×610` elements |
| solved elements | `402600` |
| padding | 150 pixels on all four sides |
| increments | 20 |
| constitutive backend | native MFront plane stress |
| MGIS threads | 8 |

The frozen selection protocol is
`validation/nonlocality_p48_preregistration.md`.

## Convergence and mechanics

- 20/20 increments converged;
- zero cutback;
- 121 global Newton iterations, seven at most per increment;
- final relative residual: `9.045e-7`;
- maximum native MFront transverse residual: `1.523e-13 MPa`;
- maximum DIC boundary error: `5.551e-17 mm`;
- reaction-balance ratio: `3.182e-12`;
- all saved arrays are finite and match their recorded SHA-256.

The maximum PEEQ is `0.11290`; the removed historical `0.2` cap would not have
been active.

## Performance

| Measure | Value |
|---|---:|
| solver elapsed time | `1332.74 s` |
| complete process wall time | `1335.97 s` |
| maximum RSS | `7,869,356 KiB` |
| constitutive time | `333.02 s` |
| tangent assembly time | `360.07 s` |
| linear solve time | `546.44 s` |
| swaps reported by GNU time | 0 |

The campaign contains the historical 2D fields and all complete 3D tensor and
plane-stress-residual fields. `validation-report.json`, `preview.png`, and the
derived DIC/FEM maps were generated without rerunning the mechanical solve.

## Reproduction

```bash
source /home/jeff/.local/share/tfel/env/env.sh
export PYTHONPATH="/home/jeff/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

.venv/bin/fem-inhouse --verbose partition \
  --input data/processed/case_study \
  --output results/reconstruction-100 \
  --count 100 \
  --padding 150 \
  --increments 20 \
  --mfront-threads 8 \
  --partition-id 48
```

