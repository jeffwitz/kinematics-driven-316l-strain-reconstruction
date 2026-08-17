# Path-memory closure check — preregistration

Registered before any run. Thresholds frozen. Negative results kept.

## Object

The monotone decline of the windowed k-NN could be the curse of
dimensionality, not the absence of closure, and the signed-window features
may have broken the magnitude symmetry of a target `|Delta gamma|`. This
is the last methodological control before the discovery path is closed:
the same windows in **magnitude-respecting form** (magnitudes of the past
shears, sign-reversal indicators as separate binary features), with three
predictors of different natures — k-NN (the reference), linear ridge, and
histogram gradient boosting (non-linear, not distance-based).

## Windows (all causally aligned, LOSO, target `|Delta gamma_n|`)

| window | features |
|---|---|
| W1m | `|tau_n|, |tau_{n-1}|, |Delta tau_n|, 1{tau_n tau_{n-1} < 0}` |
| W1mg | + `|Delta gamma_{n-1}|` |
| W2m | two steps of magnitudes and reversal indicators |

Baseline: `(|tau_n|, Gamma_{n-1})`.

## Frozen reading

* If **any** predictor on **any** window reaches `R^2 >= 0.10`: the closure
  is reopened — the k-NN metric, not the data, was the limitation, and the
  phase-space path deserves a deeper model.
* If **all** predictors stay below `0.10`: the closure is declared real
  and the discovery path is closed definitively, with this as the recorded
  evidence.

## Outputs

`validation/_generated/shared_tensor_generator/path_memory_closure_check.json`
and this file's results companion.
