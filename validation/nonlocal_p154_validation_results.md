# P154 coupled-plasticity validation results

Date: 2026-07-25

This report records the pre-registered 20-increment validation sweep on P154.
All comparisons use the raw mechanical solution on the retained
`180 x 155` element core. No Helmholtz post-filter is applied to the FEM EVM.

## Frozen campaign

| Property | Value |
|---|---:|
| solved grid | `436 x 411 = 179,196` elements |
| padding | `128 pixels = 0.23552 mm` |
| length | `ell = 0.05888 mm = 32 pixels` |
| padding/length | `4.0` |
| increments | `20` |
| MFront threads | `8` |
| Href | `6547.530616602937 MPa` |

The local reference and all three positive candidates converged for all 20
increments with zero cutbacks. The coupling sweep therefore compares
converged mechanical states rather than partially completed calculations.

## Performance and convergence

| alpha | Hchi (MPa) | solver (s) | wall time | peak RSS (KiB) | Newton | nonlocal iterations | maximum per trial |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0, local | 0 | 793.98 | not measured | not measured | 119 | 0 | 0 |
| 0.5 | 3273.765308 | 1453.77 | 24:16 | 4,360,068 | 162 | 509 | 9 |
| 1 | 6547.530617 | 1680.46 | 28:02.68 | 4,321,584 | 192 | 529 | 9 |
| 2 | 13095.061233 | 1867.20 | 31:09.37 | 4,325,048 | 221 | 573 | 10 |

The process peak is effectively constant across the positive candidates at
about 4.12--4.16 GiB. Runtime increases with coupling strength because the
number of global Newton trials and micromorphic evaluations increases.
Helmholtz itself accounts for only 19.00, 21.22, and 22.58 seconds for
`alpha=0.5`, `1`, and `2`; MFront and the repeated mechanical assemblies
dominate the added cost.

All candidates had:

- zero cutbacks and zero coupling failures;
- maximum Helmholtz residual below \(8.7\,10^{-13}\);
- maximum Gauss-point plane-stress residual below
  \(1.3\,10^{-13}\) MPa;
- finite historical, complete-tensor, and micromorphic fields.

## Raw FEM--DIC metrics

The local baseline has Pearson correlation `0.403451`, relative L2 error
`0.749778`, top-10% IoU `0.255908`, and absolute DIC-q90 IoU `0.233027`.

| Registered change from local | alpha=0.5 | alpha=1 | alpha=2 | required |
|---|---:|---:|---:|---:|
| correlation gain | +0.06579 | +0.10808 | +0.16434 | at least +0.05 |
| relative-L2 reduction | 20.09% | 30.41% | 42.17% | at least 5% |
| top-10% IoU gain | +0.01343 | +0.01777 | +0.03307 | at least +0.02 |
| absolute DIC-q90 IoU gain | +0.03071 | +0.05015 | +0.07216 | at least +0.02 |
| predicted DIC-q90 active fraction | 22.97% | 22.68% | 21.85% | 5% to 20% |
| displacement relative-L2 change | -4.42% | -5.71% | -5.08% | at most +5% |

`alpha=0.5` and `alpha=1` fail the top-10% IoU and active-area criteria.
`alpha=2` passes seven of eight checks, including top-10% IoU, but its active
area remains 1.85 percentage points above the pre-registered upper bound.
Moving that bound after seeing the result would invalidate the confirmatory
contract.

## Plastic-zone diffusivity

PEEQ is assessed only as an internal localization field; its amplitude is not
compared directly with DIC EVM.

| Metric | local | alpha=0.5 | alpha=1 | alpha=2 |
|---|---:|---:|---:|---:|
| gradient RMS | 0.68148 | 0.53627 | 0.47986 | 0.42281 |
| total variation | 0.04848 | 0.03995 | 0.03564 | 0.03056 |
| peak ratio to local | 1.000 | 0.56673 | 0.45364 | 0.35474 |
| standard-deviation ratio to local | 1.000 | 0.82488 | 0.73726 | 0.63750 |

The monotonic reduction in gradient, total variation, peak, and standard
deviation demonstrates that the energetic coupling redistributes the plastic
zone in the intended direction. This does not by itself validate the material
interpretation of the length.

## Decision

The allowed conclusion is **coupled spatial interaction partially
supported**.

The energetic term substantially improves every field-error and overlap
metric, preserves displacement agreement, and maintains plane stress.
However, no candidate satisfies every pre-registered criterion. Consequently:

- no `Hchi` is frozen as an identified or transferable parameter;
- `alpha=2` is retained as the best tested diagnostic candidate;
- no P42/P48 transfer should be presented as confirmatory until the
  active-area discrepancy and selection protocol are revisited independently;
- neither the `20%` threshold nor the tested alpha range is changed
  retrospectively.

## Reduced 3D-condensed verification

The real MFront tests were replayed after the campaign:

```text
test_micromorphic_3d_condensation_matches_native_plane_stress
test_mfront_3d_condensed_tangent_matches_finite_differences
test_zero_micromorphic_coupling_reproduces_local_newton_solution
```

All three passed. This verifies the architectural path needed for a future
three-dimensional constitutive law, independently of the P154 scientific
decision.

## Saved evidence

- local: `results/constitutive-local-p0154-pad128`
- `alpha=0.5`: `results/constitutive-nonlocal-p0154-pad128-a050`
- `alpha=1`: `results/constitutive-nonlocal-p0154-pad128-a100`
- `alpha=2`: `results/constitutive-nonlocal-p0154-pad128-a200`
- each positive campaign contains `validation-vs-local.json`
- each measured positive campaign has a sibling `resource-usage.txt`

The coupled manifest hashes are, respectively:

```text
alpha=0.5  55537ff45986cda2b1687c0def446356935d3b32937d7dc450f191ee6c3db073
alpha=1    0b833eb915731f511a4d283949bbf19029500fe7a0354412c01492caa5a956b6
alpha=2    4384d460205b1af0118fe158debea4cedf776bc00b9393f7ae9819e62d6f2569
```
