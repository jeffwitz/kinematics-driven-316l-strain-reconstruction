# Article-sized MFront partition 42

This immutable campaign preserves the held-out partition used to confirm the
Helmholtz length selected on partition 48.

## Scope

| Property | Value |
|---|---:|
| global ROI | `3600×3100` elements |
| layout | `10×10`, 100 partitions |
| partition | `42`, index `(4,2)` |
| retained core | `360×310` elements |
| solved region | `660×610` elements |
| solved elements | `402600` |
| padding | 150 pixels on all four sides |
| increments | 20 |
| constitutive backend | native MFront plane stress |
| MGIS threads | 8 |

The frozen confirmation protocol is
`validation/nonlocality_p42_confirmation_preregistration.md`.

## Convergence and mechanics

- 20/20 increments converged;
- zero cutback;
- 135 global Newton iterations, seven at most per increment;
- final relative residual: `3.054e-10`;
- maximum native MFront transverse residual: `1.344e-13 MPa`;
- maximum DIC boundary error: `5.551e-17 mm`;
- reaction-balance ratio: `5.060e-14`;
- all arrays are finite and match their recorded SHA-256.

The maximum PEEQ is `0.10763`; the removed historical `0.2` cap would not have
been active.

## Performance

| Measure | Value |
|---|---:|
| solver elapsed time | `1481.35 s` |
| complete process wall time | `1484.55 s` |
| maximum RSS | `8,079,896 KiB` |
| constitutive time | `365.78 s` |
| tangent assembly time | `410.37 s` |
| linear solve time | `602.55 s` |
| swaps reported by GNU time | 0 |

The first foreground attempt received an external `SIGTERM` with exit status
143 before writing a partial partition. Its log is preserved as
`attempt1-interrupted.log`. The successful calculation was rerun unchanged in
an isolated transient user service so that the monitoring session could not
terminate it.

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
  --partition-id 42
```

