# Shared slip-law ladder — results

Against `validation/slip_law_ladder_preregistration.md`, thresholds frozen
before the runs. The gauge test passes; the ladder peaks at the system's own
history; latent hardening is not observed.

## The ladder (LOSO R², pooled over systems, 400k system-samples)

| rung | features | L2 gauge | temporal gauge |
|---|---|---|---|
| S1 | `tau^alpha` | 0.109 | 0.121 |
| S2 | `tau^alpha, Gamma^alpha` | **0.161** | **0.184** |
| S3 | + `Gamma^{beta != alpha}` | 0.133 | 0.158 |
| S5 | + all twelve `Gamma` | 0.044 | 0.061 |

## Verdict against the frozen bars

| bar | registered | measured | |
|---|---|---|---|
| gauge stability (L2 vs time-regularised, ±0.05) | | max diff **0.025** | pass |
| slip space carries the structure (S2 ≥ 0.30) | | **0.16–0.18** | fail |
| latent hardening (jump ≥ 0.10 from S2) | | **−0.03 to −0.12** | fail |
| crystallographic invariance (shared ≥ 0.8 × best per-system) | | **0.88–0.90** | pass |

## What the ladder says

1. **The gauge is not the structure.** The L2 and the time-regularised
   decompositions support the same ladder within 0.025 — the laws the data
   support do not depend on which admissible representative the
   decomposition chose. The structure is imposed by the experimental
   tensors, not by the pseudo-inverse.
2. **One shared law suffices for the twelve systems.** The pooled law
   reaches 88–90 % of the best per-system score: no system needs its own
   function — the crystallographic equivalence survives the reconstruction.
3. **The system's own history is the best rung — and the only useful
   one.** `Gamma^alpha` adds ~0.05 over `tau^alpha` alone (the causal
   variable, the history *before* the increment). The histories of the
   *other* systems do not control `alpha`: S3 and S5 degrade the score
   (kNN dilution). **Latent hardening is not observed in these data** —
   the registered reading is the negative one, and it is kept.
4. **The ceiling is ~0.18, against 0.76 in-sample.** The strong
   `tau -> gamma` structure (Spearman 0.76) survives the held-out test
   only partially: the relation drifts across increments, and neither the
   scalar `Gamma^alpha` nor the twelve together capture the drift. The
   missing hardening state is not a scalar slip accumulation — it is what
   a Méric/SRIX-type law would call an evolving internal hardening
   variable per system (or, again, the closure content of the effective
   field).

## Conclusion

The slip phase space is the right space — gauge-stable, system-invariant,
and its driving force organises the activity better than anything before —
but the held-out structure stops at `R^2 ~ 0.18` with the tested history
variables. The next discriminator is now precise: replace the scalar
`Gamma^alpha` by an *evolving* per-system hardening state (the kind
`r^alpha` in Méric-Cailletaud or the SRIX internal variables track),
computed from these same trajectories, and rerun this ladder.
