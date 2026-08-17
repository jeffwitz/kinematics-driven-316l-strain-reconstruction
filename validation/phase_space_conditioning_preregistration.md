# Continuous conditioning of the response by the local state — preregistration

Registered before any run. Thresholds frozen. Negative results kept.

## The reframed question

The geometry figures show continuous structure (the `p_eq -> Delta p` fan,
`R^2_cond ~ 0.25`) but no discrete regimes. The question is no longer
"what are the clusters?" but:

```text
What is the minimal state dimension that makes the response
deterministically predictable enough, measured on held-out increments?
```

## Procedure

A k-nearest-neighbour local estimator (k = 50, standardised features, scipy
cKDTree) — deliberately not a network. Two held-out protocols:

* **LOSO** (leave-one-state-out): train on 19 increments, predict the 20th —
  the strong test, the one that speaks;
* **random split** (50/50 on points): the weak, upper-bound reference.

## Targets and feature ladders

* Amplitude target: `log Delta p` (the fan is multiplicative); reported
  also on the raw scale.
* Direction target: `Delta theta = wrap(theta_n - theta_s)`, predicted by
  the circular mean of the neighbours; J2 gives exactly 0.

| ladder | features | question |
|---|---|---|
| A | `sigma_eq`, `p_eq` | minimal hardening law |
| B | A + `p`, `(sin, cos)` deviatoric angle | full stress state |
| C | B + max Schmid | + orientation summary |
| D | C + `(sin, cos)` of the three Euler angles | + full orientation |

## Frozen bars

* Amplitude: the state is declared **sufficient for the amplitude** if the
  LOSO `R^2(log Delta p) >= 0.5`.
* Direction: the state is declared **sufficient for the direction** if the
  LOSO circular `R^2 >= 0.5` (residual mean resultant length against the
  global one).
* The feature addition that raises the LOSO score the most is the
  empirical minimal-dimension discovery, registered as such.

## Outputs

`validation/_generated/shared_tensor_generator/phase_conditioning.json`
plus the direction panels: hexbin `Delta theta` vs `sigma_eq` per `p_eq`
quantile and vs the max Schmid factor.
