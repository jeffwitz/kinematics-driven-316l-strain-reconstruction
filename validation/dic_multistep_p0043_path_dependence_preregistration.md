# P43 final-state path dependence on PEEQ — preregistration

Date: 2026-07-30
Written before any comparison figure or metric has been computed.

## Question

Both calculations end on the **same** prescribed boundary displacement: the
multistep workflow overwrites the last history state with the prepared final
field, so the endpoint boundary data is bit-identical between the measured and
proportional runs. Any interior difference at the final state is therefore
**path dependence of the elastoplastic solution**, not a difference in what was
imposed.

How large is it on PEEQ, and is it structured or diffuse?

## Fields compared

| Label | Path | Increments | Provenance |
|---|---|---:|---|
| A | measured 40-state DIC history | 40 | `reference_data/dic_multistep_predictor_fix_p0043_v1/PEEQ.npy` |
| B | proportional ramp | 40 | to be produced, `--mode proportional`, same workflow |
| C | proportional ramp | 20 | `results/constitutive-local-p0043-pad150/partitions/0043/PEEQ.npy` |

**B is mandatory.** Comparing A against C alone would confound path dependence
with increment-count discretisation, since the archived campaign uses 20
increments. B is produced with the identical configuration to A except that the
boundary history is replaced by a proportional ramp to the same endpoint.

All metrics are evaluated on the **core only**, `core_bounds = [1440, 1800,
930, 1240]`, excluding the 150-element padding, which is not a trusted part of
the solution.

## Registered metrics

Computed for the pair A-B, and identically for the control pair B-C:

- relative L2 difference of PEEQ over the core;
- maximum absolute difference and its location;
- mean, median and 99th percentile PEEQ of each field;
- signed mean difference, to record whether the measured path accumulates more
  or less plasticity;
- Pearson correlation of the two PEEQ fields;
- intersection over union of the top-10 % PEEQ sets;
- fraction of core elements where the two paths disagree on whether PEEQ
  exceeds `1e-4`.

## Registered interpretation thresholds

Applied to the relative L2 of A against B on the core:

| Range | Conclusion |
|---|---|
| `< 5 %` | path dependence negligible at this resolution |
| `5 %` to `20 %` | path dependence present but not dominant |
| `> 20 %` | path dependence material; any identification from a single path is suspect |

## Registered control and its veto

The discretisation control B against C uses the same metrics. **If the B-C
relative L2 is not at least three times smaller than the A-B relative L2, the
path-dependence conclusion is withdrawn**: the observed difference could not
then be separated from increment-count sensitivity.

## Registered noise caveat

`dic_boundary_loading_subspace_p0043_results.md` measured 5 of 40 affine strain
increments below unit signal-to-noise, and estimated the accumulated plastic
bias from DIC noise at about `3.6 %` of the final EVM RMS. PEEQ accumulates
monotonically, so this bias is one-sided and inflates A.

**A difference in the negligible band cannot be attributed to physical
non-proportionality**, because the noise ratchet alone predicts a difference of
that order. Only a difference well above it can be, and even then the noise
contribution is not subtracted.

The registered discriminator between a noise ratchet and genuine
non-proportionality is spatial structure:

- a noise-driven excess is **diffuse**, roughly uniform over the core, and
  shows no preference for the localisation bands;
- genuine non-proportionality **concentrates** where plasticity localises.

This is measured as the ratio of the mean signed difference inside the top-10 %
PEEQ set to the mean signed difference outside it. A ratio near `1` indicates a
diffuse, noise-like excess; a ratio well above `1` indicates band-structured
path dependence. No threshold is registered for this ratio: it is reported as a
descriptive discriminator, not a test.

## Claim boundary

Whatever the outcome, this compares two **computed** fields under the same
constitutive model and the same endpoint. It says nothing about which path is
closer to the specimen's real history, since no force synchronisation and no
unloading branch exist. PEEQ is not a DIC observable and is never compared with
a measured field here.

## Deliverable

`validation/dic_multistep_p0043_path_dependence_results.md` and
`reference_data/dic_multistep_path_dependence_p0043_v1/`.
