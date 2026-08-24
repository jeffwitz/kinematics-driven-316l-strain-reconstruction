# E-SRIX-FEMU-PATH-002R — preregistration

Date: 2026-08-24

## Objective

Test convergence of the corrected direct FEMU sensitivity with respect to the
discrete loading path. The authoritative level L0 is the 94-step corrected
common path from `srix_femu_common_path_gate_v17`.

## Fixed choices

- M8 SRIX twin, corrected Dirichlet initialization contract 2;
- preset parameters and `eta = log(tau0, R, Q, b)` unchanged;
- oracle tolerances, Newton, line-search, plane-stress and MFront settings
  unchanged from v17;
- one nonlinear base forward and one direct sensitivity Jacobian per level;
- no new finite-difference forwards at refined levels;
- same normalized physical observation fractions at L0, L1 and L2;
- no identification and no P43.

## Nested levels

- L0: v17 path, 94 steps;
- L1: mandatory midpoint of every L0 interval, with local strict repairs if
  needed;
- L2: mandatory midpoint of every L1 interval, with the same repair policy.

Mandatory midpoints are never removed. Local repairs may add nodes but may not
change the boundary history or the oracle configuration.

## Primary gate: L1 to L2

- observed forward relative L2 change `< 5e-3`;
- relative change of columns 1--3 `< 2e-2`;
- column cosines 1--3 `> 0.999`;
- rank-3 principal angle `< 2 degrees`;
- first three normalized singular values change by `< 5%`.

The fourth mode is diagnostic only. Record `sigma4/sigma1` and the alignment
of its right singular vector with `(0, 0, 1, -1)/sqrt(2)`. Do not use its
relative change as a blocking criterion.

## Artefacts

Create `validation/reference_data/srix_femu_path_convergence_v3/` containing
the report, compressed forward/Jacobian arrays, nested path fractions and a
figure. The report must record actual step counts, mandatory refinement level,
local repair count, interval widths, timing and claims. Identification and P43
remain unauthorized regardless of the outcome.
