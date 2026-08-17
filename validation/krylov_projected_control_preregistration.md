# The projected-Krylov control: raw modes, free coefficients, thermodynamics after assembly

Registered before any run. Thresholds frozen. Negative results kept.

## Why this control exists

The raw Krylov basis already showed the strongest DIC fit in the family table
(`0.602 / 0.392 / 0.245` held out at r = 4/8/16 against J2's `0.895 / 0.876 /
0.855`) — at the price of ~42-44 % negative plastic power and a stress
alignment `chi ~ 0.02`. Its weakness was therefore not expressivity but
thermodynamics. The branch-D construction exists precisely for this: take the
**raw** modes `Phi_K`, free signed coefficients `a in R^r`, form
`v = Phi_K a` first, and only then project the **assembled** field onto the
dissipative half-space — never `sum_k a_k P_H(phi_k)`. This is the last purely
geometric control registered before crystallography, and it is run before any
further network investment.

The projector does not prescribe the J2 direction; it only removes the
anti-dissipative component:

```text
P_H^{G_p}(v) = v + ReLU(-sigma^T v) / (sigma^T G_p^{-1} sigma) * G_p^{-1} sigma
```

with the plastic gauge `G_p = 2/3 [[2,1,0],[1,2,0],[0,0,1]]`, so the correction
is minimal in the true plastic-tensor metric. The Euclidean projector is kept
as an ablation; the primary result is the `G_p` one.

## The registered trap

`P_H` does **not** remove the zero-work directions `sigma : Delta eps^p = 0`,
and raw Krylov had `chi ~ 0` — its fit lived largely tangent to the stress.
The projector lets those directions pass untouched, so a technically
dissipative but physically suspect solution (huge plastic amplitude nearly
orthogonal to `sigma`) is the pre-declared failure mode. `E`, `chi`, `p_eq`
and the work split must therefore be read together; the thermodynamic
half-space alone is necessary, not sufficient.

## Two Krylov constructions, never mixed

1. **Oracle** — the basis is built from the residuals of *all* states,
   including the held-out ones. It measures expressivity only: how far
   mechanically adapted directions plus dissipation alone can go. Not
   predictive.
2. **Predictive** — the basis is built from the training states only (the
   construction of the existing `krylov` baseline). This is the line
   comparable to the generator on the held-out states.

## The fit

For each state, sequentially, with the predictor accumulating the projected
increments of the previous states:

```text
min_a  1/2 | A * P_H^{G_p}(Phi_K a) - g |^2,      a in R^r signed and free
```

with an exact gradient (A is linear with a qualified exact adjoint; the
projection transpose is exact on the frozen active set). Optimiser: L-BFGS-B,
multi-start from zero, the unprojected least-squares solution and one random
start; the best objective is kept. r = 4, 8, 16. Window: P43 100×100 first —
the reference table lives there — then the 200×200 window if the result
justifies it.

**Equivalence check before the experiment:** with the projector disabled, the
predictive fit must reproduce the archived raw-Krylov scores
(`0.602 / 0.392 / 0.245` held out). A mismatch means the replication is
wrong, not the physics.

## Frozen decision criteria (100×100, held-out states)

The line `Krylov + P_H^{G_p}` is declared **strategy-changing** only if, at
r = 16, all three hold:

1. `E <= 0.59 - margin = 0.386` (one frozen 100×100 margin below the raw
   Krylov r=16 score of `0.392` is not the bar — the bar is one margin below
   the **J2 family's best**, `0.855`, i.e. `<= 0.651`; the stronger reading
   `E <= 0.386` is reported as the expressivity ceiling). **Decision:**
   `E(r=16) <= 0.651` is the registered bar for "useful"; `E(r=16) <= 0.386`
   is the registered bar for "strategy-changing".
2. `chi_global` does not collapse below the raw-Krylov level (`~0.02-0.04`);
   registered threshold: `chi_global >= 0.02`.
3. `p_eq` (accumulated equivalent plastic strain) stays within a factor two
   of the raw-Krylov value at the same rank (`3.65e-3` at r=4 in the archived
   smoke); registered threshold: `p_eq <= 2 * p_eq_raw(rank)`.

The failure signature is pre-declared: `E` small with `chi -> 0` and/or
`p_eq` exploding means the projected Krylov fit lives in the tangent-to-stress
directions, and the control is reported as such — dissipative in the letter,
not in the spirit.

## Reported per rank, per state and globally

`E` (per state, held-out median), `chi` (mean where active, and work-weighted
global), `p_eq` (accumulated), `D_-` (negative-work share, expected zero by
construction against the predictor stress), `D_+` (positive work), projection
activity, and the multi-start spread.

## Out of scope

No network. No new twin. No kernel analysis. No RID. No crystal plasticity.
