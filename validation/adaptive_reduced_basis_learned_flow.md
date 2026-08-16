# Learning the plastic flow direction: results

P43, 100x100, states 21 to 40 against state 20. **Temporal** holdout: increments
24, 28, 32, 36 and 40 receive no gradient; the rollout passes through them on
the model's own predictions. Fifteen increments train. The metric is the share
of the elastic defect surviving,
`|eps_measured - eps_sim| / |eps_measured - eps_elastic|`, averaged over the
held-out increments -- 1.0 means no better than pure elasticity, 0 means the
defect is gone.

Startup gates, both asserted every run: adjoint dot product below 1e-8
(measured 1.5e-15), elastic lifting interior residual below 1e-10 relative
(measured 6e-12). The constrained solves are checked against independent
solvers: NNLS against `lsq_linear` at 1.3e-9, the inequality QP against a
reference SLSQP at 1e-12.

## The table

| basis | r=4 | r=8 | r=16 | negative power | chi | net D |
|---|---|---|---|---|---|---|
| krylov, fixed | **0.6021** | **0.3916** | **0.2450** | 42-44 % | +0.016 to +0.033 | 2.0e3 |
| J2 aligned, free sign | 0.8947 | 0.8762 | 0.8547 | 26-28 % | +0.39 to +0.41 | 1.0-1.4e4 |
| learned, unconstrained | 0.6077 | 0.5470 | 0.506 (partial) | 37-43 % | not measured | -- |
| learned, projected, `a >= 0` | 0.6209 | 0.6511 | 0.5869 | 8-11 % | +0.31 to +0.36 | 1.2-1.4e4 |

`chi = (sigma : d eps_p) / (|sigma| |d eps_p|_F)` over plastically active
points, a genuine cosine: the denominator carries the three-dimensional
Frobenius norm `sqrt(z^T M_p z)`, not `p_eq`, which would inflate it by
`sqrt(3/2)`. The unconstrained learned run at r=16 was stopped at step 126 to
repair the plastic gauge and never restarted; its last reading is quoted as
partial.

## What is established

**Fitting quality and physical plausibility are anticorrelated here, sharply.**
Krylov wins every fitting column and is the least physical object in the table:
44 % of its plastic power flows backwards and its increments are almost exactly
orthogonal to the stress, `chi = +0.016` at rank 16. The J2 arm is the best
aligned and the worst fitting. The learned constrained basis sits in between on
both axes, and is the only one that is decent on both.

**Krylov's rank buys cancellation, not plastic work.** Its net dissipation is
2.04e3, 2.00e3 and 2.05e3 at ranks 4, 8 and 16 -- flat -- while the absolute
power roughly doubles and `p_eq` grows from 3.65e-3 to 6.90e-3. The extra modes
add equal and opposite plasticity that sums to nothing energetically and buys
DIC agreement. The constrained learned basis does six times more net plastic
work, 1.2 to 1.4e4.

**Learning the direction beats prescribing it, by a wide margin.** Against the
hand-built J2 arm at equal rank: 0.62 against 0.89, 0.65 against 0.88, 0.59
against 0.85. The measurement that motivated this -- that isotropic normality is
not the direction the specimen takes -- is confirmed from the other side.

**Enforcing dissipation costs little at low rank and much at high rank.** From
unconstrained to constrained: +0.013 at r=4, +0.104 at r=8. It is not that the
constraint bites harder with rank; it is that what the rank was buying is
precisely what the constraint forbids.

**The constrained arm still improves with rank, and an earlier claim of a
plateau was premature.** 0.6209, 0.6511, 0.5869 -- noisy, non-monotone, but r=16
is the best of the three. The improvement is real and far weaker than the
unconstrained 0.608 to 0.506.

**The escape route did not open.** The half-space forbids `sigma . v < 0` but
permits `sigma . v = 0` with an enormous `v`, so the network could have traded
anti-dissipative work for huge nearly orthogonal increments. It did the
opposite: `p_eq` stays at 8.2 to 8.8e-3 across ranks while `chi` *rises* with
rank, +0.310 to +0.356. A three-step smoke test had suggested the reverse and
was too small to mean anything.

**No mode collapse.** The smallest over largest singular value of `A Phi` is
0.51, 0.32 and 0.23 at ranks 4, 8 and 16. The nominal rank is the effective
rank, so nothing about the rank behaviour is an artefact of colinearity.

## The cone result, which closes an old question

The minimal thermodynamic requirement is on the final combination alone,
`C a >= 0` with `C_g = sigma_g^T Phi_g` and `a` free in sign -- weaker than
projecting every mode and then forcing non-negative coefficients. Testing it on
**unprojected** modes is impossible: the largest `|a|` attainable inside
`{C a >= 0}` is **0.0000**, at random initialisation and after training, at
ranks 4 and 8 alike, by a feasibility LP.

Twenty thousand half-spaces through the origin of `R^4` leave nothing unless
their normals are strongly clustered -- that is, unless the modes are already
nearly all dissipative. So the projection is not an extra restriction laid over
the physics; with globally constant coefficients it is close to the only way the
physics admits anything at all. This also settles the question left open since
the Krylov campaign: that cone was genuinely trivial, not a QP declining to
leave the origin.

What *was* excessive is `a_k >= 0`. With projected modes the feasible set
contains the positive orthant and more, since a negative coefficient is
admissible wherever the others compensate pointwise. That arm is running.

## Known limits of this campaign

* **The temporal holdout is interpolation.** Withholding every fourth increment
  along a smooth loading path is a test a basis spanning the neighbouring
  residuals passes almost by construction, and Krylov reaches 0.0169 on the
  fitting region at rank 16 -- it memorises the trajectory. The extrapolating
  split, training on 21-33 and testing on 34-40, is the honest instrument and
  has not been run.
* **Dissipation is verified at the midpoint but enforced against the predictor
  stress.** The residual 8 to 11 % of negative power is that gap, and it
  shrinks with rank. `D_-` against the predictor should be numerically zero and
  is not separately reported yet.
* **The projection is Euclidean in the plane coordinates**, not minimal in the
  physical plastic metric `G_p`. The consistent form is
  `v + relu(-sigma^T v)/(sigma^T G_p^-1 sigma) G_p^-1 sigma`. Both guarantee
  `sigma^T v >= 0`; the difference is an ablation, not a repair.
* The unconstrained r=16 run never finished, and the unconstrained arm predates
  both the alignment diagnostics and weight saving, so it cannot be re-scored.
