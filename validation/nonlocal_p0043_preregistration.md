# P43 coupled-J2 campaign preregistration

Date: 2026-07-26

## Scientific scope

P43 is the selected 10 × 10 calibration ROI because visual inspection shows
two distinct diagonal deformation bands. This campaign evaluates coupling
strength at the previously retained diagnostic length. It does not identify a
new length and does not modify the constitutive formulation after seeing the
results.

| Property | Frozen value |
|---|---:|
| partition | P43, index `(4, 3)` |
| core bounds | `x=[1440,1800)`, `y=[930,1240)` |
| core shape | `360 × 310 = 111,600` elements |
| padding | 150 pixels |
| solved shape | `660 × 610 = 402,600` elements |
| pixel size | 1.84 µm |
| length | 58.88 µm = 0.05888 mm = 32 pixels |
| padding/length | 4.6875 |
| increments | 20 |
| MFront threads | 8 |
| backend | native MFront plane stress |
| linear system | upper CSR, PARDISO `mtype=2` |

The reference local campaign is
`results/constitutive-local-p0043-pad150`. It converged all 20 increments in
129 Newton iterations without cutback. External elapsed time was `943.31 s`
and peak RSS was `4,288,740 KiB`.

## Frozen coupling values

`estimate-nonlocal-reference` selects the 87,519 core elements with
`PEEQ > 1e-6`, or `78.4220%` of the core. The registered median of
`K*n*PEEQ**(n-1)` is:

```text
H_ref = 5168.147582748343 MPa
```

The sweep is fixed before any coupled result is inspected:

| alpha | Hchi (MPa) | campaign |
|---:|---:|---|
| 0 | 0 | `results/constitutive-local-p0043-pad150` |
| 1 | 5168.147582748343 | `results/constitutive-nonlocal-p0043-pad150-a100` |
| 2 | 10336.295165496686 | `results/constitutive-nonlocal-p0043-pad150-a200` |
| 4 | 20672.59033099337 | `results/constitutive-nonlocal-p0043-pad150-a400` |

The source reference is
`results/constitutive-local-p0043-pad150/HREF.json`, whose local campaign
manifest SHA-256 is
`81dbec648ace9012ae5898b24b1e50d58dab12ea73e906521c286f8d43ce4429`.

## Frozen numerical settings

```text
nonlocal relaxation          = 0.5
relative coupling tolerance  = 1e-6
maximum coupling iterations  = 15
maximum Helmholtz residual   = 1e-10
Newton settings              = unchanged project defaults
```

All three coupled runs use the same DIC input, mesh, padding, length,
material maps, increments, tolerances, thread count, and executable commit.
Only `Hchi` changes.

## Evaluation

Metrics are computed on the core only. Primary comparison uses raw historical
EVM reconstructed from DIC and FEM displacement with the same operator. No
Helmholtz post-filter is applied to primary FEM EVM.

The existing prospective criteria are retained:

- correlation gain at least `0.05`;
- relative L2 reduction at least `5%`;
- top-10% IoU gain at least `0.02`;
- DIC-q90 absolute-threshold IoU gain at least `0.02`;
- q90 active area between `5%` and `20%`;
- interior displacement error degradation no greater than `5%`;
- finite fields, valid plane stress, and no abnormal cutback accumulation.

PEEQ is interpreted only as an internal plasticity field, never as an
experimental observable. The figures use common colour scales. If the
response improves monotonically through `alpha=4`, no larger alpha is launched
or selected without a new preregistration.

