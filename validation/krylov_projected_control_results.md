# Projected-Krylov control — results

Against `validation/krylov_projected_control_preregistration.md`, thresholds
frozen before the runs. Equivalence passed; the trap did not occur; the
strategy-changing bar is missed by 0.016.

> **Correction, recorded after the run and before any conclusion.** The
> plumbing check requested before the experiment (FD sweep of the projected
> chain gradient) caught an algebraic error in the transpose of the `G_p`
> projector: the dual was corrected along `G_p^{-1} sigma` instead of
> `sigma`. The Euclidean branch was exact (dot test `1.8e-11`, which is why
> the Euclidean numbers below stand as measured); the `G_p` branch failed the
> dot test at 3.4 % and the sweep flatlined at 18 %. After the fix
> (`a540fd8`) the sweep is V-shaped with a `1.6e-10` minimum. All `G_p` lines
> were recomputed with the corrected gradient: the optima shifted by at most
> `0.002` on `E`, and no verdict, criterion or finding changed. The
> previously committed `G_p` JSONs are superseded by the `_fixed` artifacts,
> kept for provenance.

## Verdict against the four frozen criteria (100×100, held-out means)

| criterion | registered | measured (corrected) | |
|---|---|---|---|
| equivalence, projector off | reproduce archived Krylov | mean `0.6021`, fitted `0.3955`, `p_eq 3.648e-3`, `chi 0.033` — exact | pass |
| 1 — useful | `E(r16) <= 0.651` | **0.4015** | pass |
| 2 — strategy-changing | `E(r16) <= 0.386` | **0.4015** | **fail by 0.016** |
| 3 — `chi_global >= 0.02` | no tangent trap | `0.864`–`0.932` | pass |
| 4 — `p_eq <= 2 x raw(rank)` | no amplitude explosion | **1.01x at every rank** | pass |

The pre-declared failure signature — small `E` with `chi -> 0` and/or
exploding `p_eq` — did not occur.

## The table (held-out means; medians beside each result JSON)

| rank | raw Krylov | +`P_H^{G_p}` | +`P_H^{euclid}` | oracle `G_p` | learned, dissipative | J2 imposed |
|---|---|---|---|---|---|---|
| 4 | 0.602 | 0.653 | 0.653 | 0.574 | 0.621 | 0.895 |
| 8 | 0.392 | 0.492 | 0.493 | 0.426 | 0.651 | 0.876 |
| 16 | 0.245 | 0.402 | 0.409 | 0.336 | 0.587 | 0.855 |

Thermodynamics of the corrected `G_p` line: negative-work share (midpoint)
`3.4 / 6.0 / 6.8 %` (raw: `43 / 46 / 47 %`), work-weighted alignment
`0.932 / 0.880 / 0.864` (raw: `0.135 / 0.086 / 0.062`), projection active on
~46 % of points, multi-start spread `<= 1e-3` relative.

## The two dissipations, measured separately

| rank | `D_-^pred` share | `D_-^mid` share | `D_+` (work) |
|---|---|---|---|
| 4 | **4.4e-17** | 3.4 % | 8.44e3 |
| 8 | **4.7e-17** | 6.0 % | 1.20e4 |
| 16 | **4.7e-17** | 6.8 % | 1.66e4 |

`D_pred = sigma_pred : Delta eps^p` is non-negative to roundoff, as `P_H`
imposes it — measured, not assumed. `D_mid = (sigma_{n-1}+sigma_n)/2 :
Delta eps^p` is the honest a posteriori diagnostic and keeps a small negative
fraction, as expected. `D_+` is the total positive work.

## Three measured facts

**1. The projector redirects, it does not suppress.** The accumulated
equivalent plastic strain of the projected fits equals the raw one to within
1 % at every rank (`3.641 vs 3.648`, `5.655 vs 5.718`, `6.943 vs 6.897`,
all `e-3`). The half-space constraint corrects the *direction* of the
plasticity at constant amplitude — this is why `chi_global` jumps from
`~0.06` to `~0.87+` while `E` loses only `0.05–0.16`.

**2. The correction metric barely matters.** The Euclidean projector scores
`0.653 / 0.493 / 0.409` against `G_p`'s `0.653 / 0.492 / 0.402` — the
half-space constraint does the work, not the metric inside it. The `G_p`
choice is retained on physical grounds, not on fit grounds.

**3. The price of predictivity is small.** The oracle line (basis built from
the held-out DIC itself — an expressivity ceiling, never a predictor) reaches
`0.336` at r=16, against `0.402` for the predictive line: the held-out
information the basis renounces is worth `0.066` at r=16.

## Provenance of the historical Krylov modes

Verified against the replication, not assumed: the archived table
(`0.6021 / 0.3916 / 0.2450`) was built from the **training states' residuals
only** (`seeds = transpose_numpy(measured[s] - elastic[s]) for s in
training`, then QR and residual growth). It is therefore the **predictive**
line, not an expressivity oracle — the equivalence check reproduces it bit
for bit from the same construction. The oracle variant measured here is the
expressivity ceiling and is labelled as such; the two are never mixed.

## What this says about the network

At equal rank, on the same window and the same temporal holdout, the fixed
projected-Krylov geometry beats the learned dissipative generator by
**0.19–0.25** at r=8/16 (0.492 vs 0.651; 0.402 vs 0.587) and matches it at
r=4 (0.653 vs 0.621), with cleaner thermodynamics (3.4–6.8 % negative power
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
* The reported `D_-^mid` is the midpoint-stress measure; `D_-^pred` is
  pointwise non-negative by construction and measured at `~5e-17`.
* All numbers are the 100×100 qualification window. The 200×200 replay is the
  registered next step if the strategy question is reopened there.
* Multi-start: three starts per state (zero, unprojected least squares, one
  random); the spread is `<= 1e-3`, so the piecewise-linear optima are
  stable.
