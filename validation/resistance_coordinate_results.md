# Resistance coordinate — results

Against `validation/resistance_coordinate_preregistration.md`, thresholds
frozen before the runs. The verdict is a clean negative, and the tuned
parameters say why.

## Verdict

| bar | registered | measured | |
|---|---|---|---|
| overstress is the coordinate | `R^2 >= 0.30` | **0.034** | fail |
| the jump is real | `>= +0.10` | **-0.002** | fail |
| parameters stable | spread `<= 0.3` | `b` spread **0.32** | fail |

Baseline `(|tau|, Gamma~) -> |Delta gamma|`: LOSO `R^2 = 0.036`.
Resistance `(xi, Gamma~) -> |Delta gamma|`: `R^2 = 0.034`.

## What the tuning itself says

The per-fold parameter search — which had the freedom to make the
resistance anything on the grid — chose **`a = 0` in every fold** (no
isotropic resistance), **`c = 0` in nearly every fold** (no latent
coupling), and an unstable small `b` (`0.3` or `1.0`, spread 0.32). Given
the chance to build an overstress coordinate, the data answer that the
resistance built on the scalar slip accumulations does not exist at this
scale: the best the grid can do is a tiny self-correction that does not
survive the held-out test.

## Reading, kept honest

The missing hardening state is **not** an isotropic resistance over the
`Gamma` histories — the simplest Méric-like form is refuted on these
trajectories. Combined with the ladder (the twelve `Gamma` diluting the
predictor) and the in-sample-to-LOSO gap (0.76 → ~0.04–0.16), the picture
is consistent: the `tau -> gamma` structure is real but almost entirely
carried by the increments themselves, and no tested static function of
`(tau, Gamma)` recovers it across states. What remains as candidates, in
the order the analyses support: an *evolving, saturating* per-system state
(not a linear accumulation), or the closure content of the effective
field — which the FCC gauge test showed is not an artefact of the
decomposition, but is also not the slip activity itself.

## Outputs

`validation/_generated/shared_tensor_generator/resistance_coordinate.json`
(folds, tuned parameters, verdicts).
