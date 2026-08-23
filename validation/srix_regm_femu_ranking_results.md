# SRIX-REGM versus FEMU ranking — Gate 5 result

Date: 2026-08-23  
Primary artefact: `reference_data/srix_regm_femu_ranking_v1/report.json`  
Pre-registration: `srix_regm_femu_ranking_preregistration.md`

## Gate result

All 20 off-truth parameter sets converged in the complete adaptive mechanical
solver. On the exact M8 target, SRIX-REGM ranks them consistently with the
full-FEMU displacement mismatch.

| Read-out | Frozen threshold | Result | Status |
|---|---:|---:|---|
| valid forward candidates | at least 15 | `20` | pass |
| Spearman correlation | at least 0.80 | `0.866` | pass |
| Pearson of log objectives | at least 0.70 | `0.878` | pass |
| common candidates in best five | at least 3 | `3` | pass |

The common best-five candidates are `lhs_03`, `lhs_04`, and `lhs_12`. The gate
passes without moving the population or thresholds.

## Cost

The median costs are `2.643 s` for REGM and `13.047 s` for complete FEMU, a
speedup of `4.94x`. This narrowly misses the indicative `5x` performance
target and therefore calls for profiling, not scientific rejection.

The forward cost is strongly parameter dependent:

| Quantity | minimum | median | maximum |
|---|---:|---:|---:|
| accepted adaptive steps | `34` | `42.5` | `468` |
| Newton iterations | `161` | `199` | `2159` |
| full-FEMU time (s) | `6.44` | `13.05` | `146.75` |

Accepted-step count and wall time correlate at `0.992`. This explains both the
much larger `124.48 s` cost of the exact truth trajectory, which required 338
accepted steps, and the sub-5x median speedup over a population containing many
easy forward solves. REGM replay cost is much less sensitive to nonlinear
global convergence.

## Scientific interpretation

This gate establishes a limited but useful claim: **before applying the DIC
measurement transfer, the first-order reconditioned equilibrium gap is a good
ranking surrogate for the complete nonlinear forward mismatch near the exact
twin**. Equality of objective values is neither observed nor required.

It does not cancel the negative Gate-4 result. Gate 4 shows that the qualified
observation transfer moves the REGM minimum away from the generating
parameters. Therefore the exact ranking result alone is insufficient to
authorize parameter identification on measured P43 data. A final, same-
population observed-space ranking must determine whether the surrogate ranking
survives transfer and measured noise. This additional gate uses no tuned
population or threshold and is required by the observed bias, not by the
outcome of this ranking.
