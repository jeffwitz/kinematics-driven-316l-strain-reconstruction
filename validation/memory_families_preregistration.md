# Minimal dynamical memory per system — preregistration

Registered before any run. Thresholds frozen. Negative results kept.

## Object

The static resistance over the scalar `Gamma` histories is refuted. The next
question, before any full SRIX/Méric law: **which minimal per-system memory
closes the phase space?** Three families, evolved causally on the true
decomposed activities (`sigma_gamma` = the standard deviation of `|Delta
gamma|` on the training subset; increments normalised by it):

```text
F0  Gamma^alpha :  z_{n+1} = z_n + delta_n                      (pure accumulation, the baseline)
F1  saturation  :  z_{n+1} = z_n + delta_n (1 - z_n / z_sat)    (accumulation with saturation)
F2  signed      :  x_{n+1} = (1 - d delta_n) x_n + delta_n sign(gamma_n)
F3  both        :  z as F1, x as F2, features (|tau|, z, x)
```

Then `(|tau^alpha|, memory) -> |Delta gamma^alpha|`, k-NN (k = 50), the
same pooled population and the same leave-one-state-out as the ladder and
the resistance test. Per fold, the family parameters are tuned on a
training subsample only (binned-variance score; grids: `z_sat` in
`{0.5, 1, 2, 4}`, `d` in `{0, 0.5, 2}`, F3 the coarse product), and the
held-out state never sees the tuning. The memory uses only the true
activities of the states *before* the increment being predicted — open-loop
state construction, the same convention as `Gamma`.

## Frozen bars

1. **The memory closes the gap.** At least one family reaches LOSO
   `R^2 >= 0.30` (the `Gamma` baseline in this metric is ~0.04).
2. **The family names the mechanism.** The best family's advantage over
   `Gamma` is `>= +0.10`; if the signed memory alone carries the gain, the
   directional/backstress memory is declared the missing state; if only
   saturation does, the evolving saturating state is.
3. **The parameters are stable** across folds (spread within one grid
   step), else the family is declared not pinned by the data.
4. All negatives kept: if no family improves over `Gamma`, the missing
   state is not a first-order local memory of this form.

## Outputs

`validation/_generated/shared_tensor_generator/memory_families.json` and
this file's results companion.
