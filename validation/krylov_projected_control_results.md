# Projected-Krylov control — results

Against `validation/krylov_projected_control_preregistration.md`, thresholds
frozen before the runs. Equivalence passed; the trap did not occur; the
strategy-changing bar is missed by 0.014.

## Verdict against the four frozen criteria (100×100, held-out means)

| criterion | registered | measured | |
|---|---|---|---|
| equivalence, projector off | reproduce archived Krylov | mean `0.6021`, fitted `0.3955`, `p_eq 3.648e-3`, `chi 0.033` — exact | pass |
| 1 — useful | `E(r16) <= 0.651` | **0.400** | pass |
| 2 — strategy-changing | `E(r16) <= 0.386` | **0.400** | **fail by 0.014** |
| 3 — `chi_global >= 0.02` | no tangent trap | `0.865`–`0.931` | pass |
| 4 — `p_eq <= 2 x raw(rank)` | no amplitude explosion | **1.00x at every rank** | pass |

The pre-declared failure signature — small `E` with `chi -> 0` and/or
exploding `p_eq` — did not occur.

## The table (held-out means; medians reported beside each result JSON)

| rank | raw Krylov | +`P_H^{G_p}` | +`P_H^{euclid}` | oracle `G_p` | learned, dissipative | J2 imposed |
|---|---|---|---|---|---|---|
| 4 | 0.602 | 0.653 | 0.653 | 0.575 | 0.621 | 0.895 |
| 8 | 0.392 | 0.492 | 0.493 | 0.426 | 0.651 | 0.876 |
| 16 | 0.245 | 0.400 | 0.409 | 0.335 | 0.587 | 0.855 |

Thermodynamics of the `G_p` line: negative-power share `3.5 / 6.0 / 6.8 %`
(raw: `43 / 46 / 47 %`), work-weighted alignment `0.931 / 0.880 / 0.865`
(raw: `0.135 / 0.086 / 0.062`), projection active on ~46 % of points,
multi-start spread `<= 1e-3` relative.

## Three measured facts

**1. The projector redirects, it does not suppress.** The accumulated
equivalent plastic strain of the projected fits equals the raw one to within
1 % at every rank (`3.685 vs 3.648`, `5.696 vs 5.718`, `6.889 vs 6.897`,
all `e-3`). The half-space constraint corrects the *direction* of the
plasticity at constant amplitude — this is why `chi_global` jumps from
`~0.06` to `~0.87+` while `E` loses only `0.05–0.16`.

**2. The correction metric barely matters.** The Euclidean projector scores
`0.653 / 0.493 / 0.409` against `G_p`'s `0.653 / 0.492 / 0.400` — the
half-space constraint does the work, not the metric inside it. The `G_p`
choice is retained on physical grounds, not on fit grounds.

**3. The price of predictivity is small.** The oracle line (basis built from
the held-out DIC itself — an expressivity ceiling, never a predictor) reaches
`0.335` at r=16, against `0.400` for the predictive line: the held-out
information the basis renounces is worth `0.065` at r=16.

## What this says about the network

At equal rank, on the same window and the same temporal holdout, the fixed
projected-Krylov geometry beats the learned dissipative generator by
**0.19–0.25** at r=8/16 (0.492 vs 0.651; 0.400 vs 0.587) and matches it at
r=4 (0.653 vs 0.621), with cleaner thermodynamics (3.5–6.8 % negative power
against 8–11 %). The generator's current role — inventing the plastic
geometry from scratch — is therefore not supported by this control: the
DIC-adapted mechanical geometry plus the thermodynamic half-space already
carries most of the gap reduction, predictively. What remains open for a
network is narrower and better posed: transferring or predicting modes across
increments, enriching the geometry beyond the Krylov span, or supplying the
state-dependent projector frame — not replacing the geometry itself.

## Registered caveats

* The archived reference table reports held-out **means**; all criteria read
  the mean, medians are reported beside (the equivalence check fixed the
  convention pre-run).
* The reported negative-power share is the honest midpoint-stress measure;
  the projection guarantees `sigma_pred^T Delta eps^p >= 0` pointwise by
  construction, and the midpoint figure is expected nonzero.
* All numbers are the 100×100 qualification window. The 200×200 replay is the
  registered next step if the strategy question is reopened there.
* Multi-start: three starts per state (zero, unprojected least squares, one
  random); the spread is `<= 1e-3`, so the piecewise-linear optima are
  stable.
