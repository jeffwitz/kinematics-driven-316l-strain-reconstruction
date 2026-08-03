# CPS4R reduced integration — qualification preregistration

Date: 2026-08-03
Written before any CPS4R–CPS4 elastoplastic comparison has been run.

## Purpose

CPS4R divides the number of constitutive integrations by four, which is the
largest available saving for the SRIX crystal law. It is currently exposed but
**not** authorised for scientific campaigns: the only proven equivalence with
CPS4 is elastic. This document freezes what the qualification must show, and
what would refute it, before the numbers exist.

The element algebra and the stabilisation are in
`docs/explanation/reduced_integration_hourglass.md` and are not restated here.

## What is already established, and therefore not under test

At `beta = 1` and constant elastic tangent, `K^{1pt} + (K^{4pt} - K^{1pt})`
collapses to `K^{4pt}` identically. The automated tests confirm this to `1e-13`
on the element operator and to `1e-10` through the solver on a **non-affine**
elastic load where the hourglass energy is genuinely nonzero.

That property does not survive yielding: the constitutive tangent softens while
the stabilisation stays built on the fixed elastic reference. After yielding the
two formulations are different elements. The question is not whether they differ
— they must — but **by how much, and whether the shipped diagnostic predicts it**.

## Hypotheses

**H1 — plastic divergence.** After yielding, CPS4R deviates from CPS4, and the
deviation grows with accumulated plastic strain.

**H2 — beta monotonicity.** Lowering `beta` softens the hourglass modes, so the
hourglass energy ratio rises and the deviation from CPS4 grows monotonically.

**H3 — diagnostic validity.** The hourglass energy ratio `r_hg` is a
*conservative* gate: a run passing the documented `1 %` warning threshold has a
CPS4-versus-CPS4R error inside the acceptance bound below. **This is the
hypothesis this campaign exists to test.** The whole diagnostic story in the
documentation rests on it, and it has never been checked.

**H4 — cost.** Constitutive time falls by about four. Total wall time falls by
less, because assembly, the linear solve and the boundary machinery are
unchanged.

## Registered test cases

Two, both small and both run to completion in this repository. No production ROI
is used and no long campaign is replayed.

**C1 — non-affine plastic J2.** Structured mesh at the campaign spacing
`0.00184 mm`, with a heterogeneous yield-stress map so that a band forms rather
than a uniform field, and a boundary displacement carrying a sinusoidal
perturbation on top of a tensile mean so that element-level strain is not
constant. Both formulations, and `beta` in `{0.1, 0.25, 0.5, 1.0}`.

**C2 — SRIX single crystal, tilted orientation.** Homogeneous orientation away
from the axes, so that the reference operator is genuinely anisotropic and the
rotated condensed cubic elasticity is exercised, at `beta = 1`. This case tests
the crystal path, not `beta`.

A case is only admissible if it excites the modes; see F1.

## Registered metrics

Against the CPS4 solution of the same case, treated as the reference:

- displacement, relative `L2` over all degrees of freedom;
- displacement, **absolute** RMS difference in mm, reported against the DIC
  boundary noise `sigma = 0.0511 px = 9.40e-5 mm`;
- equivalent plastic strain, relative `L2` over elements, and maximum absolute;
- reaction force, relative `L2`;
- hourglass energy ratio `r_hg`, and the spatial field;
- cutbacks and total Newton iterations;
- `elapsed_seconds` and `constitutive_seconds`.

## Registered acceptance criteria

Derived, not chosen by taste. The project already accepts a displacement-field
reproduction error of `1.673 %` for its archived reference profile. An element
formulation must not become a comparable error term. Requiring that it inflate
that error by no more than `10 %` in quadrature gives
`sqrt(1.673^2 + x^2) <= 1.1 * 1.673`, hence `x <= 0.766 %`.

**A1** — equivalent plastic strain, relative `L2` difference `<= 0.5 %`
(the bound above, rounded down).

**A2** — displacement, relative `L2` difference `<= 0.1 %`. Displacement is the
smoother field and the one the reconstruction reports; it is held an order of
magnitude tighter than PEEQ.

**A3** — the absolute RMS displacement difference is reported against `sigma`.
Below `sigma` the two formulations are not distinguishable by the measurement
this project performs, and the choice between them is scientifically free. This
is a reported fact, not a pass condition.

**A4** — `r_hg` below the documented `1 %` warning threshold.

**A5** — CPS4R introduces no cutback that CPS4 did not have.

A value of `beta` is **recommended** only if A1, A2, A4 and A5 all hold. If
several qualify, the largest is recommended, because `beta = 1` is the only
value with an exact elastic equivalence and departing from it buys nothing
unless the hourglass modes are over-stiffened.

## Registered falsifiers

**F1 — case validity.** If `r_hg < 1e-8` at `beta = 1`, the case does not excite
the hourglass modes, every `beta` will look identical, and **no conclusion about
`beta` may be drawn from it**. The case is then declared inadmissible and
redesigned; the failure is recorded, not silently replaced. This is exactly the
trap the existing affine tests fell into.

**F2 — monotonicity.** If the deviation from CPS4 is not monotone in `beta`,
H2 is false, `beta` cannot be selected by this procedure, and **no value is
recommended**.

**F3 — diagnostic validity.** If any run has `r_hg < 1 %` while its PEEQ
relative `L2` error exceeds `0.5 %`, the ratio is **not** a conservative gate.
H3 is then refuted and the `1 % / 5 %` thresholds must be withdrawn from
`docs/explanation/reduced_integration_hourglass.md` and from the scientific
contract, or restated as descriptive only. This outcome is publishable and will
be published.

**F4 — cost.** If the total wall-time saving at equal accuracy is below `1.5x`,
CPS4R does not repay the qualification burden on this problem class and is not
recommended, whatever the accuracy result. The constitutive-time saving alone
does not satisfy this criterion.

## What this campaign cannot conclude

It uses two small synthetic cases. A pass authorises CPS4R for **cases of this
kind** — structured mesh, plane stress, Dirichlet-driven, no micromorphic
coupling. It does not authorise it on an experimental ROI, and it does not
authorise the combination with the nonlocal extension, which stays refused.

Whatever the outcome, CPS4 remains the reference formulation and the default.

## Recording

Results go to `validation/cps4r_qualification_results.md`, including negative
and refuting outcomes. The generating script is
`scripts/qualify_reduced_integration.py` and its JSON output is archived beside
the report.
