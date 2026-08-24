# Sequential one-correction SRIX diagnostic

This twin-only diagnostic tests whether the missing REGM/FEMU sensitivity is
caused by the absence of causal displacement--internal-state feedback. At each
M8 increment, the previous accepted corrected state is advanced by the twin
increment. One consistent tangent correction is computed, the trial is
reverted, SRIX is re-evaluated at the corrected displacement, and the state is
committed. No global Newton solve is performed.

The primary artifact is
`validation/reference_data/srix_regm_sequential_one_newton_v2/report.json`.

| quantity | sequential one-correction | observed FEMU |
|---|---:|---:|
| normalized singular values | `1, 0.56251, 0.05764, 2.30e-4` | `1, 0.54152, 0.40668, 0.06787` |
| condition number | `4.35e3` | `14.7` |
| leading principal angle | `45.24 deg` | reference subspace |
| rank-2 principal angles | `67.91 deg, 12.38 deg` | reference subspace |

The sequential feedback substantially raises the second singular direction
relative to fixed-history REGM (`0.422`) and raises the weakest direction from
`4.65e-5` to `2.30e-4`. It nevertheless leaves the rank-two subspace almost
as misaligned with FEMU as before (`67.9 deg` versus `67.2 deg` for fixed
REGM), and the two weak FEMU directions remain far smaller (`0.0576, 2.3e-4`
versus `0.4067, 0.0679`).

## Decision

The one-correction causal replay is an informative improvement but **does not
pass the surrogate-geometry gate**. The missing FEMU information is not
recovered by changing only the reconditioner or by one approximate feedback
step. This closes the planned low-cost REGM diagnostic sequence: do not launch
P43 identification. A future method would need a validated tangent sensitivity
or a deliberately reduced objective, followed by a fresh twin/FEMU ranking
gate; it must not be presented as SRIX parameter identification yet.

```{figure} _static/evidence/srix_regm_sequential_one_newton.png
:alt: Information geometry for sequential one-correction SRIX replay and observed FEMU.
:width: 95%

The causal correction improves conditioning and one sensitivity direction, but
does not reproduce the observed FEMU sensitivity subspace.
```

## Observable audit

The v2 diagnostic scores the correction of the last increment. A follow-up
rerun also scores the cumulative accepted endpoint displacement, which is the
quantity directly comparable to an endpoint FEMU sensitivity. The corrected
comparison is in
`validation/srix_regm_sequential_one_newton_cumulative_results.md`; it remains
negative and does not authorize P43.
