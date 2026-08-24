# Sequential SRIX replay: cumulative endpoint correction

This twin-only rerun tests a methodological correction to the diagnostic in
`srix_regm_sequential_one_newton_v2`. The previous script scored only the last
one-step Newton correction. The new run retains that observable, but also
scores the accepted endpoint displacement gap

```text
accepted[n] - displacement_history[n]
```

before applying the same affine-preserving transfer. The latter is the
quantity comparable to a FEMU displacement sensitivity at a scored endpoint.

No mechanical campaign, P43 run, or parameter identification was performed.
The calculation uses the M8 twin, the registered eight endpoints, the same
log-parameter central difference `h = 3e-3`, and the same wrap-free transfer.
The previous v2 artifact is preserved unchanged.

## Result

| observable | normalized singular values | condition number | rank-1 angle to FEMU | rank-2 angles to FEMU |
|---|---:|---:|---:|---:|
| last correction only | `1, 0.56251, 0.05764, 2.30e-4` | `4.35e3` | `45.24°` | `67.91°, 12.38°` |
| cumulative endpoint gap | `1, 0.46460, 0.09381, 2.17e-4` | `4.60e3` | `74.67°` | `65.78°, 11.01°` |
| observed FEMU reference | `1, 0.54152, 0.40668, 0.06787` | `14.7` | — | — |

The cumulative observable raises the third direction from `0.0576` to
`0.0938`, but it does not recover the FEMU weak directions: the FEMU values
are `0.4067` and `0.0679`, while the cumulative fourth direction remains
`2.17e-4`. Its leading subspace is also less aligned with FEMU than the
last-correction observable (`74.67°` versus `45.24°`).

For reference, the cosine of the four raw columns with the FEMU columns is:

| column | last correction | cumulative endpoint gap |
|---:|---:|---:|
| `log_tau0` | `-0.0358` | `0.1321` |
| `log_R` | `0.1401` | `0.3370` |
| `log_Q` | `0.3724` | `0.0002` |
| `log_b` | `-0.2557` | `0.6364` |

These raw-column cosines are not a substitute for the SVD comparison (the
parameter columns are correlated), but they confirm that replacing the
observable is not sufficient to reproduce the FEMU Jacobian.

## Decision

The proposed scoring correction is valid and has been tested. It changes the
reported geometry, but it does **not** turn the one-correction sequential
replay into a FEMU-equivalent sensitivity surrogate on M8. The earlier
negative conclusion about this particular surrogate therefore stands, with a
more complete observable audit:

* scoring only the last correction was methodologically incomplete;
* scoring the cumulative accepted endpoint gap is the correct comparison;
* the corrected cumulative result still fails the FEMU geometry gate.

P43 remains blocked. The next permitted development is the separately
registered direct FEMU sensitivity method with persistent constitutive shadow
histories; no P43 identification should be launched from this diagnostic.

Primary machine-readable artifact:
`validation/reference_data/srix_regm_sequential_one_newton_v3/report.json`.

The generated figure is
`validation/reference_data/srix_regm_sequential_one_newton_v3/srix_regm_information_geometry.png`.
