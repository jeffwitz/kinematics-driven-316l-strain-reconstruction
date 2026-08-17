# Phase-space analysis of the reconstructed inelastic trajectories — preregistration

Registered before any run. Thresholds frozen. Negative results kept.

## Object

The projected-Krylov control produced admissible inelastic trajectories on
the P43 window. This analysis asks the question that precedes any
constitutive identification: **is there a local law at all, on the domain of
states the test actually visits?**

```text
S_n(x) ~ S_m(y)  ==>?  Delta eps^inel_n(x) ~ Delta eps^inel_m(y)
```

The trajectories are saved per point and per increment
(`sigma`, `eps^e`, `eps^inel`, `Delta eps^inel`, and the observable-projected
versions) by the control script's `--trajectories` mode. All primary
statements below are computed on the **observable-projected** increments
(the nullspace projection of the displacement operator, the 19 measured
directions); the raw fields are reported beside them, never instead of them.
Without this projection a perfectly local material can look non-local, and
this analysis exists to test locality, so it must not inherit the kernel.

## The three quantities

1. **Direction dispersion conditional on state.** The state is binned on the
   in-plane deviatoric geometry and the accumulated equivalent plastic
   strain: quantile bins on `|s|` (8), the deviatoric angle in the Kelvin
   plane (8), and `p_eq` (4), minimum 50 points per bin. Within each bin,
   the circular standard deviation of the angle between the increment's
   deviatoric direction and `s` is reported, and aggregated over the best
   populated decile of bins.

2. **Amplitude structure.** The same bins: the ratio of the within-bin
   variance of the increment gauge amplitude to its global variance gives
   `R^2_cond(Delta p | S) = 1 - mean_bin(Var_in-bin) / Var_global`.

3. **Coverage of the visited domain `Omega_P43`.** The number of populated
   bins, the share of points in the best decile, and the participation ratio
   of the bin counts — the honest measure of how much of the constitutive
   space one tensile path actually visits.

## Frozen readings (asymmetric by construction)

* **Clean direction structure** — the best-decile circular standard
  deviation of the angle to `s` is `<= 15 degrees` — supports a local flow
  structure on `Omega_P43`. It does **not** uniquely identify the law: a
  single near-monotone path cannot separate J2 from any family of laws that
  coincide on the path. This is registered as a support statement, nothing
  more.
* **Non-J2 but stable direction** — the same dispersion is `<= 15 degrees`
  while the mean angle to `s` in the best bins is `>= 20 degrees` — is the
  strongest possible outcome: an associative flow direction different from
  J2, observed directly in the data.
* **J2-associated** — dispersion `<= 15 degrees` and mean angle `<= 10
  degrees`: associativity with `s` is supported on the visited domain.
* **Dirty structure** — dispersion `> 15 degrees` in well-populated bins on
  the observable-projected increments — is a strong finding in the other
  direction: missing internal variables (orientation, history, gradients) or
  a reconstruction closure, and it says so with the kernel excluded.
* `R^2_cond(Delta p | S) >= 0.5` is the registered bar for "the amplitude is
  structured by the local state"; anything below is reported as amplitude
  non-locality on the visited domain.

## What is explicitly not claimed

No constitutive law is identified by this analysis. Nothing about 316L
beyond the visited window. No transferability. One test visits one path:
coverage numbers are the guard against over-reading a single experiment.

## Inputs and outputs

Input: `validation/_generated/shared_tensor_generator/krylov_trajectories.r16.npz`
(from the predictive projected-Krylov line, rank 16). Output:
`validation/_generated/shared_tensor_generator/phase_space_analysis.json` and
this file's results companion `phase_space_local_law_results.md`.
