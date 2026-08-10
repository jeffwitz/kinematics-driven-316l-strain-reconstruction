# SRIX P43 M200 isolated-failure diagnostics

This diagnostic replays the exact EBSD crop `[1520:1720] x [985:1185]`
with the historical unsmoothed SRIX law, `mfront_threads=4`, Krylov BLAS
threads set to one, and the qualified composite tangent enabled.  It captures
only single-point MGIS trials that fail during the GPS bisection probe; the
nominal constitutive algorithm is unchanged.

The run completed with 58 global Newton iterations and 978 substepped points.
It recorded 380 isolated failed trials on 50 distinct points.  The raw
committed/trial MGIS rows are in:

```text
validation/_generated/performance/
  srix_p43_m200_failure_diagnostics.failure_diagnostics.npz
  srix_p43_m200_failure_diagnostics.analysis.json
```

The post-processing reconstructs the SRIX overstress

```text
z_s = |tau_s - X_s| - r_s
```

from the trial stress and `dg`, and uses the committed `p` and `a` values for
the hardening radius and backstress.

| diagnostic | result |
| --- | ---: |
| isolated failed trials | 380 |
| distinct points | 50 |
| minimum `|z_s|` | `2.57e-3 MPa` |
| `|z_s| < 1e-2 MPa` | 2 components |
| `|z_s| < 1e-1 MPa` | 6 components |
| `|z_s| < 1 MPa` | 91 components |
| `|dg_s| < 1e-12` | 2846 of 4560 components |
| negative `dg_s` | 844 of 4560 components |
| active systems in captured trials | 1–4 |

The current evidence does not support the claim that the dominant failure
mechanism is simply the Macaulay threshold at `z=0`: almost no captured
failure is exactly at that threshold.  In contrast, zero or sign-changing
`dg` branches are common.  This points toward the absolute-value/sign terms
in the hardening and backstrain update as the next diagnostic target.

The local Newton iteration counter is deliberately reported as unavailable for
failed trials: the generated auxiliary counter is promoted only on successful
integration and remains zero when MGIS returns failure.  Obtaining the actual
iteration count requires a lower-level MFront/TFEL hook; this diagnostic does
not infer it from the zero value.

No compact Charbonnier transition has been introduced yet.  The next
constitutive experiment should therefore target the `dg` sign/absolute-value
branches, or instrument them directly, before another smoothing sweep.
