# P154 coupled-plasticity smoke results

Date: 2026-07-25

These runs use the `p0154-smoke` profile: a `308 x 283` solved grid,
64 pixels of padding, five nominal increments, and eight MGIS threads. The
padding is only twice the registered length. These results validate the
implementation and select the first full-size candidate; they are not
scientific evidence for the internal-length hypothesis.

## Convergence-norm diagnostic

The first `alpha=0.5` run used the original global mixed \(L_2\) stopping
criterion. It completed and all fields were preserved, but required:

- `1330.18 s`;
- 30 converged subincrements and 19 cutbacks;
- 171 mechanical Newton iterations;
- 1198 micromorphic iterations;
- 19 fixed-point failures before cutback.

This revealed the mesh-size dependence documented in the numerical amendment
to `nonlocal_p154_preregistration.md`. No positive-coupling DIC metric was
inspected before the amendment. The candidate runs below use the frozen
`mixed_relative_linf` criterion.

## Corrected smoke sweep

The mechanically local reference is the coupled behaviour with \(H_\chi=0\).
It shares the exact mesh, increment schedule, MFront behaviour family, and
output contract with the positive candidates.

| alpha | Hchi (MPa) | elapsed (s) | cutbacks | Newton | nonlocal iterations | final nonlocal residual |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 201.15 | 3 | 66 | 63 | 0 |
| 0.5 | 3273.765308 | 406.28 | 3 | 71 | 353 | \(1.9622\,10^{-8}\) |
| 1 | 6547.530617 | 503.04 | 2 | 84 | 441 | \(2.5700\,10^{-8}\) |
| 2 | 13095.061233 | 226.30 | 0 | 46 | 173 | \(1.7892\,10^{-8}\) |

All positive candidates had zero fixed-point failures with the corrected
norm. For `alpha=2`, the maximum Helmholtz residual was
\(1.3098\,10^{-12}\), and the maximum Gauss-point plane-stress residual was
\(1.1448\,10^{-13}\) MPa. The lower elapsed time of this particular smoke
run follows from its absence of cutbacks and lower number of global Newton
trials; it is not interpreted as monotonic performance scaling with \(H_\chi\).

## Raw FEM-DIC comparison on the core

The validator reconstructs `EVM_HISTORICAL` independently from the DIC and FEM
displacements. It applies no post-filter and evaluates only the
`180 x 155` core.

| Metric | local, alpha=0 | alpha=0.5 | alpha=1 | alpha=2 |
|---|---:|---:|---:|---:|
| Pearson correlation | 0.366580 | 0.434710 | 0.481387 | 0.547805 |
| relative L2 | 0.635581 | 0.518425 | 0.455124 | 0.381023 |
| top-10% IoU | 0.229888 | 0.253933 | 0.264159 | 0.278351 |
| absolute DIC-q90 IoU | 0.210687 | 0.240086 | 0.261117 | 0.296064 |
| DIC-q90 predicted active fraction | 19.885% | 18.806% | 17.749% | 15.968% |
| displacement relative L2 | 0.001250 | 0.001148 | 0.001087 | 0.001023 |

Relative to the local smoke result:

| Registered gain | alpha=0.5 | alpha=1 | alpha=2 |
|---|---:|---:|---:|
| correlation | +0.06813 | +0.11481 | +0.18123 |
| relative-L2 reduction | 18.43% | 28.39% | 40.05% |
| top-10% IoU | +0.02404 | +0.03427 | +0.04846 |
| absolute DIC-q90 IoU | +0.02940 | +0.05043 | +0.08538 |
| displacement-error change | -8.19% | -13.03% | -18.15% |

All candidates pass every smoke-level acceptance check. `alpha=1` improved
every registered metric over `alpha=0.5`, but missed two criteria on the
validation profile. Because the best result lay at the upper boundary of the
initial registered interval, the pre-registration allowed the additional
`alpha=2` smoke. It again improved every registered metric and was therefore
promoted to the 128-pixel, 20-increment validation profile. The alpha value is
not frozen for transfer until that validation-profile comparison is complete.

## Saved evidence

- `results/constitutive-nonlocal-p0154-smoke-a000`
- `results/constitutive-nonlocal-p0154-smoke-a050` (original L2 diagnostic)
- `results/constitutive-nonlocal-p0154-smoke-a050-linf`
- `results/constitutive-nonlocal-p0154-smoke-a100-linf`
- `results/constitutive-nonlocal-p0154-smoke-a200-linf`
- each corrected positive campaign contains `validation-vs-a000.json`

Every calculation directory contains its manifest, status, hashes, historical
fields, complete stress/strain tensors, micromorphic fields, and solver
diagnostics.
