# Observed-space SRIX-REGM versus FEMU ranking — STOP/GO result

Date: 2026-08-23  
Primary artefact:
`reference_data/srix_regm_femu_observed_ranking_v1/report.json`  
Pre-registration: `srix_regm_femu_observed_ranking_preregistration.md`

## Decision

**NO-GO for P43 parameter identification with the present SRIX-REGM
observation formulation.** All 20 forward simulations converged, but the
transfer-only level fails every ranking threshold. The frozen decision required
both observation levels to pass.

| Level | Spearman | log-Pearson | top-5 overlap | Decision |
|---|---:|---:|---:|---|
| T1 transfer | `0.326` | `0.276` | `2/5` | fail |
| T2 transfer + noise + whitening | `0.866` | `0.861` | `4/5` | pass |

Thresholds were respectively `0.80`, `0.70`, and `3/5`. The T1 p-values are
`0.160` and `0.240`; its weak correlations are not distinguishable from zero
on this population.

## Why the noisy pass does not reverse the decision

In T2, the whitened FEMU RMS lies only between `0.870324` and `0.870559`; its
coefficient of variation across parameters is `9.1e-5`. The measured-noise
realization therefore dominates the absolute FEMU objective almost entirely.
The good rank correlation concerns very small parameter-dependent differences
on top of that common noise floor.

By contrast, T1 contains no random displacement and already exposes the
structural mismatch. Its REGM RMS ranges from `1.899e-7` to `2.432e-7 mm`,
while observed FEMU ranges from `1.186e-8` to `7.370e-8 mm`, but their rankings
do not agree. Adding a dominant noise term cannot validate a surrogate whose
noiseless observed ranking is wrong.

The REGM rankings at T1 and T2 are nearly identical (Spearman `0.9985`), while
the FEMU rankings change strongly (Spearman `0.377`). This is consistent with
the Gate-4 diagnosis: the current `O(delta_u)` residual and the full observed
forward displacement mismatch react differently to the measurement chain.

## What remains valid

- The weak equilibrium-gap implementation is mechanically and transactionally
  verified.
- Exact-space SRIX-REGM recovers the generating twin parameters.
- Exact-space ranking against full FEMU passes (`0.866` Spearman, `0.878`
  log-Pearson, `3/5` overlap).
- The method remains useful as a diagnostic or a proposal generator in exact
  kinematic space.
- It is not an admissible identification objective for measured P43 data in
  its current observed-space form.

## Consequence

No P43-A or P43-M100 optimization is launched, and no SRIX parameter is
reported as identified. The next scientific task is not more optimization. It
is to reformulate the observation-aware equilibrium discrepancy so that a
mechanically equilibrated digital twin remains centred on its generating
parameters and candidate rankings remain correlated with observed FEMU.

Possible future work must be pre-registered and re-enter through the same twin
gates. Examples include comparing equilibrium only in modes demonstrably
transmitted by DIC, or defining the likelihood of the measured kinematics
without treating a spatially filtered displacement as an exact equilibrium
trajectory. A sequentially reconditioned solver is not justified until this
observation inconsistency is resolved.
