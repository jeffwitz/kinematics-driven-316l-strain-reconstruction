# Continuous conditioning — results

Against `validation/phase_space_conditioning_preregistration.md`, thresholds
frozen before the runs. Both bars fail; the ladder says what the next
variable must be.

## Verdict

| target | ladder | LOSO score | bar | |
|---|---|---|---|---|
| amplitude `log Delta p` | A `(sigma_eq, p_eq)` | **0.130** | 0.5 | fail |
| | B + full stress | 0.105 | 0.5 | fail |
| | C + Schmid | 0.104 | 0.5 | fail |
| | D + Euler | 0.103 | 0.5 | fail |
| direction `Delta theta` | A | -0.075 | 0.5 | fail |
| | D (best case) | **0.116** | 0.5 | fail |

Random-split reference (weak upper bound): amplitude `R^2 ~ 0.16-0.18`
across ladders; direction still `<= 0.15` with `MAE ~ 90 deg` — the
no-information level for a wrapped angle.

## Three measured facts

1. **The amplitude carries continuous structure — weak and state-specific.**
   `(sigma_eq, p_eq)` is the best predictor at every protocol (in-sample fan
   `R^2_cond ~ 0.25`, random split ~0.17, LOSO **0.13**). The
   in-sample-to-LOSO gap is the finding: part of the conditioning is
   temporal drift across increments that `p_eq` alone does not capture —
   the history variable is real but incomplete.
2. **Orientation adds nothing to the amplitude** (0.130 → 0.103 across the
   ladder) — the max Schmid factor and the full Euler angles organise the
   *states* (clustering AMI 0.98) without ever conditioning the *response*.
3. **The direction of the effective field is not conditioned by any tested
   state, continuously.** The circular `R^2` stays at or below 0.12 with a
   90° mean error even with full orientation: the flow direction is
   statistically independent of `(stress, p_eq, orientation)` — with the
   registered caveat that the thermodynamic projector constrains the *full*
   field, so the observable part alone is not required to be dissipative.

## The minimal-dimension answer

No tested state renders the response deterministic enough, but the ladder
locates the deficit: the amplitude needs a **history/temporal variable
richer than the scalar `p_eq`** (the LOSO gap), and the direction needs
something no local state tested here contains — the prime suspect being the
closure content of the effective field, to be separated by the
`Delta eps_D / Delta eps_0` decomposition, after which the same analysis
reruns on `Delta eps_D` alone.

## Direction panels

`validation/_generated/shared_tensor_generator/phase_geometry_direction_dtheta.png`
(wrapped `Delta theta` vs `sigma_eq` per `p_eq` quantile) and
`phase_geometry_dtheta_schmid.png`: no region of the visited domain shows a
reproducible flow direction — the panels are flat at the uniform level,
consistent with the circular scores.

## Corrected synthesis

The geometry figures and this analysis together replace the earlier
over-strong statement. What is demonstrated is: **the effective inelastic
field does not admit discrete constitutive regimes, and no tested local
state determines its direction — but a continuous, non-trivial amplitude
geometry exists** (the `p_eq -> Delta p` fan, a quarter of the variance
in-sample, a tenth held-out), with the history variable as the measured
missing piece.
