# The resistance coordinate — preregistration

Registered before any run. Thresholds frozen. Negative results kept.

## Object

The slip-law ladder says the raw `Gamma` histories are not the hardening
variable. The natural next probe, before any full Méric/SRIX law: build the
simplest internal resistance and ask whether the *overstress* is the phase
coordinate the data want.

```text
r^alpha_n   = tau_ref (a + b Gamma~^alpha_n + c Gamma~^{beta != alpha}_n),
             Gamma~ = Gamma / median(Gamma),    tau_ref = median |tau|
xi^alpha_n  = |tau^alpha_n| - r^alpha_n
(xi^alpha_n, Gamma~^alpha_{n-1})  ->  |Delta gamma^alpha_n|
```

Three dimensionless parameters `(a, b, c)` — an isotropic resistance
(`a`), a self-hardening (`b`) and a latent-hardening (`c`) coefficient. No
FEM equilibrium anywhere.

## Protocol

* Same pooled population as the ladder (one random system per point-state
  sample, 20 000 per state, L2 decomposition).
* **Per leave-one-state-out fold**, the parameters are tuned on a 100 000
  subsample of the 19 training states only — a 4x4x4 grid over
  `{0, 0.3, 1, 3}` with a binned-variance score (10 quantile bins on
  `xi`) — then the k-NN (k = 50) evaluation runs on the full training fold
  with the tuned parameters, and the held-out state is predicted. The
  held-out state never sees the tuning.
* Baseline per fold, same population, same k-NN: `(|tau|, Gamma~^alpha)`
  — the S2 rung in magnitude form.

## Frozen bars

1. **The overstress is the coordinate.** The LOSO `R^2(|Delta gamma|)` of
   the resistance model reaches `>= 0.30` (the S2 ceiling was 0.18).
2. **The jump is real.** `R^2(resistance) - R^2(baseline) >= +0.10`.
3. **The resistance parameters are stable** across folds: the inter-fold
   spread of the tuned `(a, b, c)` is reported; a spread larger than the
   grid step (0.3) means the data do not pin the resistance — a named
   negative.
4. Negative outcomes kept: if the jump is below 0.10, the resistance
   coordinate is declared insufficient and the missing state is not an
   isotropic-kinematic resistance of this form.

## Outputs

`validation/_generated/shared_tensor_generator/resistance_coordinate.json`
and this file's results companion.
