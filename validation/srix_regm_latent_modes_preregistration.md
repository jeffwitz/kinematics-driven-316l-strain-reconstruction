# Preregistration — latent kinematic modes for SRIX-REGM

Date: 2026-08-24

This is a twin-only diagnostic following the observation-placement ablation.
It reuses the M8 exact twin, the same 20 parameter candidates and the same
observed full-FEMU ranking. No new nonlinear forward solve is allowed.

## Construction

Let `u*` be the archived exact mechanical history and let `u_obs=O(u*)` be the
qualified affine-preserving transfer. Form the snapshot matrix of the missing
kinematics `D = u* - u_obs`, one row per replayed state, and compute its SVD.
The right singular vectors are spatial displacement modes. For rank `k`:

```text
u_k = u_obs + projection_rank_k(D)
```

The constitutive replay receives `u_k`; the pseudo-displacement is scored with
the affine-preserving observation transfer. `k=0` is the observed-input path;
the full rank reconstructs the exact twin history.

## Frozen read-outs

Report `k = 0, 1, 2, 3, 4, 5` and the numerical full rank. For each rank report:

- cumulative POD energy of `D`;
- SRIX-REGM RMS at the truth;
- Spearman and log-Pearson against the frozen observed FEMU ranking;
- best-five overlap;
- no new FEMU cost.

The experiment is descriptive. A rank that improves the ranking does not yet
define a latent reconstruction for real DIC, because the POD basis uses the
known twin truth. No P43 identification is authorized.
