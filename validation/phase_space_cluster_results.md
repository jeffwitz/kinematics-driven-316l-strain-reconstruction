# Phase-space clustering — results

Against `validation/phase_space_cluster_preregistration.md`, thresholds
frozen before the runs. The frozen outcome is the named one: **the tested
variables do not determine the response** — and the sharpest-variable
comparison answers which variable matters most.

> **Recorded deviation.** The DBCV-equivalent validity index was not
> computed: its quadratic cost is prohibitive at the 40 000-point subsample.
> Silhouette is reported instead, as the preregistration's first named index.
> The HDBSCAN parameter amendment (mcs 50 / ms 5, recorded pre-reading) is in
> the preregistration.

## Verdict against the frozen bars

| feature set | clusters | noise | AMI across ranks (bar 0.5) | time-mixed | direction gain (bar 1.4) | R²_amp (bar 0.5) |
|---|---|---|---|---|---|---|
| F1 stress | 2–3 | 0 % | 0.45 | ✓ | 1.00 | 0.00 |
| F2 +p_eq | 2 | 0 % | **0.12** | ✓ | 1.00 | 0.02–0.04 |
| F3 +Schmid | 2 | 0 % | **0.67** | ✓ | 1.00 | 0.00 |
| F4 +Euler | 11–12 | 8 % | **0.98** | ✓ | 1.00–1.01 | 0.07 |

Kernel exclusion (raw vs observable `p_eq`, F2): AMI **0.15** (bar 0.5) —
the kernel carries cluster structure in the hardening coordinate.

**No feature set passes a response-conditioning bar.** The clusters are
real, stable and time-mixed structures of the *state* — and they carry no
information about the *response*: within-cluster direction dispersion equals
the global isotropic one (gain 1.00) and the amplitude stays unstructured
(R² ≤ 0.14). The frozen reading applies verbatim: *the variables do not
determine the response — the missing-variable search continues.*

## What the bars say, one by one

1. **`p_eq` destabilises.** Adding the reconstructed hardening level to the
   state makes the labeling reconstruction-specific (AMI 0.12 across ranks,
   0.15 raw-vs-observable). The hardening field of the admissible
   reconstruction is not a stable constitutive coordinate at this stage.
2. **Orientation is the sharpest variable — for stability, not for the
   response.** The Schmid factor restores robustness (0.67) and the full
   Euler features give the most reproducible labeling (0.98, 11–12
   clusters, 8 % noise): the grain/orientation structure is the strongest
   state-space organisation the data contain. But the clusters it defines
   do not condition the flow direction or the amplitude.
3. **The response remains isotropic inside every family.** Even within
   orientation-defined clusters, the deviatoric direction of the increment
   is as dispersed as globally. A regime law `F_k(S)` built on these
   clusters would explain nothing.

## Conclusion, in the terms of the registered pipeline

The admissible effective inelastic field does not behave like a local
constitutive response of `(stress, hardening, orientation)` — the closure
content (3D effects, unmodeled physics, reconstruction) dominates the
response even inside the most reproducible state families. This closes the
regime-law path on the *raw effective field* and makes the registered next
step the decisive one: the `Delta eps_D / Delta eps_0` decomposition, with
the same clustering and conditioning analysis applied to the dissipative
component alone. If a law structure exists in these data, it lives in
`Delta eps_D` — not in the field as reconstructed.

## Caveats

* Exploratory HDBSCAN parameters (amended pre-reading), low silhouettes —
  the structures found are density contrasts on a smooth path, not sharp
  regimes; the frozen bars, not the cluster count, are what spoke.
* One window, one loading path: the missing-variable search continues on
  richer descriptors (tensorial history, gradients) and on the 200×200
  window where the same analysis will multiply the sample by four.
* The kernel exclusion bar was applied to F2 only, as registered.
