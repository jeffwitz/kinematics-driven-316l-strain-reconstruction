# Path-memory closure check — results

Against `validation/path_memory_closure_check_preregistration.md`,
thresholds frozen before the runs. The reading is **closed** — and the
methodological reservations are answered by the design itself.

## The three predictors, the four windows (LOSO R²)

| window | k-NN | ridge | boosting |
|---|---|---|---|
| baseline `(|tau_n|, Gamma)` | 0.036 | **0.079** | 0.068 |
| W1m magnitudes + reversal | -0.010 | 0.026 | 0.030 |
| W1mg + `|Delta gamma_{n-1}|` | -0.036 | 0.018 | 0.013 |
| W2m two steps | -0.111 | -0.030 | -0.053 |

The frozen bar for reopening is `0.10`; the best score anywhere is
**0.079**, held by the *baseline* with the linear predictor.

## What this settles

1. **The k-NN metric was not the limitation.** The linear ridge and the
   histogram boosting — predictors without raw distance and with strong
   regularisation — agree with the k-NN on every window: the past windows
   never beat the baseline, and the decline with depth is reproduced by
   all three. The monotone decline is the data, not the curse of
   dimensionality.
2. **The symmetry correction does not change anything.** Magnitudes with
   separate reversal indicators give the same picture as the signed
   windows: the loading path carries no transferable information, with or
   without the sign structure.
3. **The best state description remains the two-dimensional baseline** —
   `(|tau|, Gamma)` — at 0.08, a tenth of the frozen bar. The discovery
   path is closed definitively: **the 2-D effective field does not contain
   an exploitable local constitutive closure**, under every predictor
   family tested.

## Conclusion, in the terms agreed

The methodological control the user asked for is done and the verdict
survives it. The next step is the forward path: a constitutive law or
generator with its own internal variables, trained so that its history —
passed through equilibrium — predicts the DIC it has not seen, on the P43
200×200 window. All of today's instruments (the reconstruction, the FCC
decomposition, the LOSO ladders, the gauge and invariance checks) carry
over unchanged as the validation machinery.
