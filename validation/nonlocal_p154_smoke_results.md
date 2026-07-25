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

Both positive candidates had zero fixed-point failures with the corrected
norm. Their maximum Helmholtz residuals were respectively
\(1.2424\,10^{-12}\) and \(1.2807\,10^{-12}\). Their maximum Gauss-point
plane-stress residuals were \(1.3849\,10^{-13}\) MPa and
\(1.9859\,10^{-13}\) MPa.

## Raw FEM-DIC comparison on the core

The validator reconstructs `EVM_HISTORICAL` independently from the DIC and FEM
displacements. It applies no post-filter and evaluates only the
`180 x 155` core.

| Metric | local, alpha=0 | alpha=0.5 | alpha=1 |
|---|---:|---:|---:|
| Pearson correlation | 0.366580 | 0.434710 | 0.481387 |
| relative L2 | 0.635581 | 0.518425 | 0.455124 |
| top-10% IoU | 0.229888 | 0.253933 | 0.264159 |
| absolute DIC-q90 IoU | 0.210687 | 0.240086 | 0.261117 |
| DIC-q90 predicted active fraction | 19.885% | 18.806% | 17.749% |
| displacement relative L2 | 0.001250 | 0.001148 | 0.001087 |

Relative to the local smoke result:

| Registered gain | alpha=0.5 | alpha=1 |
|---|---:|---:|
| correlation | +0.06813 | +0.11481 |
| relative-L2 reduction | 18.43% | 28.39% |
| top-10% IoU | +0.02404 | +0.03427 |
| absolute DIC-q90 IoU | +0.02940 | +0.05043 |
| displacement-error change | -8.19% | -13.03% |

Both candidates pass every smoke-level acceptance check. `alpha=1` improves
every registered DIC metric more than `alpha=0.5`, so it is launched first on
the 128-pixel, 20-increment validation profile. The alpha value is not frozen
for transfer until the validation-profile comparison is complete.

## Saved evidence

- `results/constitutive-nonlocal-p0154-smoke-a000`
- `results/constitutive-nonlocal-p0154-smoke-a050` (original L2 diagnostic)
- `results/constitutive-nonlocal-p0154-smoke-a050-linf`
- `results/constitutive-nonlocal-p0154-smoke-a100-linf`
- each corrected positive campaign contains `validation-vs-a000.json`

Every calculation directory contains its manifest, status, hashes, historical
fields, complete stress/strain tensors, micromorphic fields, and solver
diagnostics.
