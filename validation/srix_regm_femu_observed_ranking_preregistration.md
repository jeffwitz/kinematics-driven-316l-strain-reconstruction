# Observed-space SRIX-REGM versus FEMU ranking preregistration

Date: 2026-08-23  
Status: **frozen after Gate 4 exposed observation bias and before any observed ranking**

## Why this additional gate is necessary

The exact-space ranking passed, but the already pre-registered transfer/noise
twin showed that the DIC observation chain moves the REGM minimum away from
the generating SRIX parameters. The exact ranking therefore cannot by itself
authorize use on measured P43 data.

This gate reuses, without changing, the 20 off-truth candidates and thresholds
from `srix_regm_femu_ranking_preregistration.md`. It asks whether REGM and full
FEMU still rank those laws consistently after both are expressed in the
qualified observation space.

## Fixed levels

1. `T1 transfer`: the exact truth and every candidate forward displacement are
   passed through the same affine-preserving DIC transfer. REGM replays the
   transferred truth history and observes its correction with the same
   transfer.
2. `T2 transfer_noise`: the target additionally contains the deterministic
   measured-noise realization frozen in Gate 4. Candidate forward predictions
   contain no artificial noise. FEMU differences and REGM corrections are
   whitened with the same qualified spectral whitener.

Only the eight macro observation endpoints contribute to FEMU. All adaptive
truth states remain in causal REGM replay. FEMU RMS is computed on interior
nodal DOFs, matching the exclusion of unknown boundary reactions.

## Frozen decision

Each level is judged separately using the existing thresholds:

- at least 15 complete candidates;
- Spearman at least `0.80`;
- log-objective Pearson at least `0.70`;
- at least three common candidates in the best-five sets.

P43 is authorized only if **both T1 and T2 pass**. If either fails, the current
REGM observation formulation is a NO-GO for P43 parameter identification. No
threshold, noise realization, transfer function, population, or parameter
range may be adjusted after the result.
