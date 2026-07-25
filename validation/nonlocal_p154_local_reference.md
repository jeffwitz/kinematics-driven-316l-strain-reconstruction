# P154 local reference for the coupled micromorphic campaign

Date: 2026-07-25  
Status: completed local reference; no coupled scientific conclusion

## Immutable scope

- prepared inputs: `data/processed/case_study`
- layout: `20 × 20`
- partition: `154`, index `(7, 14)`
- retained core: `x=[1260,1440)`, `y=[2170,2325)`
- solved bounds: `x=[1132,1568)`, `y=[2042,2453)`
- solved grid: `436 × 411 = 179196` elements
- padding: `128` pixels
- backend: `mfront-native-plane-stress`
- increments: `20`
- MGIS threads: `8`
- coupling: disabled

## Reproduction command

```bash
source /home/jeff/.local/share/tfel/env/env.sh
.venv/bin/fem-inhouse --verbose partition \
  --input data/processed/case_study \
  --output results/constitutive-local-p0154-pad128 \
  --parts-x 20 \
  --parts-y 20 \
  --padding 128 \
  --increments 20 \
  --constitutive-backend mfront-native-plane-stress \
  --partition-id 154 \
  --mfront-threads 8
```

## Solver result

| Quantity | Value |
|---|---:|
| elapsed solver time | `793.980136784 s` |
| converged increments | `20 / 20` |
| attempted increments | `20` |
| cutbacks | `0` |
| total Newton iterations | `119` |
| maximum Newton iterations | `7` |
| final relative residual | `2.146828468e-08` |
| maximum Gauss-point plane-stress residual | `1.204658830e-13 MPa` |
| constitutive time | `183.607363900 s` |
| tangent assembly time | `191.821960957 s` |
| linear solve time | `368.058288178 s` |

Campaign manifest SHA-256:
`8193f94cfe15124b3dfd5bd2522e099cec645cfe54ad7f143bc238b21a0dd35c`.

The complete arrays and `status.json` remain under
`results/constitutive-local-p0154-pad128`. The status records one SHA-256 for
every saved array; `PEEQ.npy` is
`12b486038a361570d19adc01c8b91c1025d1398ae95d5d83e648e83c531dec87`.

## Reference coupling modulus

The command

```bash
.venv/bin/fem-inhouse estimate-nonlocal-reference \
  --input data/processed/case_study \
  --campaign results/constitutive-local-p0154-pad128 \
  --partition-id 154 \
  --output results/constitutive-local-p0154-pad128/HREF.json \
  --alphas 0 0.25 0.5 1 2
```

selects core elements with `PEEQ > 1e-6` and evaluates
`K * n * PEEQ**(n-1)`.

| Quantity | Value |
|---|---:|
| core elements | `27900` |
| plastic core elements | `24507` (`87.8387 %`) |
| derivative minimum | `361.685668 MPa` |
| derivative q25 | `3682.372248 MPa` |
| derivative median, `H_ref` | `6547.530617 MPa` |
| derivative q75 | `13398.837089 MPa` |
| derivative maximum | `6625241.387737 MPa` |

The pre-registered candidates are:

| alpha | `H_chi` |
|---:|---:|
| `0` | `0 MPa` |
| `0.25` | `1636.882654 MPa` |
| `0.5` | `3273.765308 MPa` |
| `1` | `6547.530617 MPa` |
| `2` | `13095.061233 MPa` |

The large upper tail is expected near the regularized origin because
`n < 1`. The campaign uses the pre-declared median, not the maximum or an
after-the-fact value.
