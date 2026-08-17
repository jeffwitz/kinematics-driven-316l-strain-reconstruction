# Minimal dynamical memory — results

Against `validation/memory_families_preregistration.md`, thresholds frozen
before the runs. A clean negative, with the tuning choosing the answers.

## Verdict

| bar | registered | measured | |
|---|---|---|---|
| the memory closes the gap | `R^2 >= 0.30` | best family **0.017** | fail |
| the jump is real | `>= +0.10` over `Gamma` | **-0.019** | fail |
| parameters stable | | yes (identical per fold) | pass, uninformatively |

| family | memory | tuned params (all folds) | LOSO R² |
|---|---|---|---|
| F0 | pure accumulation `Gamma` | — | **+0.036** |
| F1 | saturating `z` | `z_sat = 0.5` everywhere | -0.003 |
| F2 | signed `x` | `d = 0` everywhere | -0.066 |
| F3 | both | `(1.0, 0.5)` everywhere | +0.017 |

## What the tuning says

Given the freedom to build any first-order dynamical memory on the grid,
the data answer that **no memory of these families helps**: the saturating
memory is worse than the plain `Gamma` (and the tuning pushes it to its
most aggressive saturation, where it still loses), the signed memory is
worse still (the tuning sets the decay to zero — the memory "wants" to be
`Gamma` even with signed dynamics available), and the combination does not
recover. The negative is registered as such: **the missing state is not a
first-order local memory of the slip activities.**

## The completed chain of negatives, read together

* the raw `Gamma` histories do not condition the activity across
  increments (ladder: S2 ceiling 0.18 signed, 0.04 magnitude);
* a static linear resistance over the `Gamma` is refused by the tuning
  (`a = 0`, `c = 0`, unstable `b`);
* a first-order saturating or signed memory is refused the same way;
* and yet the gauge test passed and the FCC representation is
  kinematically exact — none of this is a decomposition artefact.

The consistent conclusion: the in-sample structure (`tau -> gamma`,
Spearman 0.76) is carried by the increments themselves — the trajectory —
and no local state recoverable from the 2-D effective field closes it. The
closure content of the effective field is not separable by any tested
variable. What this points to is not another static state hypothesis but
one of two registered paths: drive the true SRIX/Méric internal variables
(closed-loop dynamics) on these trajectories and test *them* as features,
or move the constitutive structure back into the forward model and use
these analyses as the validation instrument — the phase space of the
effective field, as reconstructed, has now been searched to the limit of
first-order states.
