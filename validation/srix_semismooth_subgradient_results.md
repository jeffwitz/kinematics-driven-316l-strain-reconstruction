# Results — `sign(0) = 0` is a subgradient correction, not a constitutive change

Protocol and thresholds: `srix_semismooth_subgradient_preregistration.md`,
frozen before each of the two runs. Executed under `7512c66`.

## Verdict

The preregistered verdict rule required P1, P2, H2 and H3. **All four hold.**
`sign(0) = 0` may be recommended as the default convention of the SRIX law,
with the historical value kept as a compatibility switch.

H1 **failed at its literal threshold** and is reported as such below. Its
underlying claim was then established by a stronger test that was not
preregistered.

## H3 — the transplant reproduces the archive

At the full increment, on the 380 archived isolated failures:

| variant | converged |
|---|---:|
| historical, `sign(0) = -1` | 0 / 380 |
| `sign(0) = 0` | 380 / 380 |
| `δ = 1e-5` | 380 / 380 |

Identical to the archived replay. The comparison rests on a faithful
transplant.

## H1 — failed as written, at `6.1e-11` against a `1e-11` threshold

I preregistered that the historical and zero-subgradient variants would agree
within `1e-11` wherever both converge. Measured, on the reduced-increment
replays where the historical convention does converge:

| `λ` | comparable | max on stress | max on any internal variable |
|---:|---:|---:|---:|
| `0.125` | 380 | `5.49e-11` | `5.63e-11` |
| `0.25` | 380 | `3.79e-11` | `6.12e-11` |
| `0.5` | 379 | `3.67e-11` | `6.01e-11` |

The threshold is exceeded by roughly half a decade. **The threshold was
mis-specified.** I set it by assuming the error on the solution would be of the
order of the residual tolerance `@Epsilon 1.e-14`; that neglects the
conditioning of the local Jacobian, which at these near-degenerate cusp states
amplifies a converged residual into a solution difference some three decades
larger.

That is an explanation, and an explanation offered after seeing the number is
worth nothing unless it makes a prediction that can fail. It does: if the two
conventions share a residual, their solutions must converge to each other as
the tolerance tightens, proportionally and without a floor. If instead the
residual differs, the difference must plateau.

### The discriminating measurement

Sweeping `epsilon` at `λ = 0.25`, all 380 records:

| `epsilon` | `sign(0) = 0` vs historical | `δ = 1e-5` vs historical |
|---:|---:|---:|
| `1e-10` | `9.94e-08` | `1.355e-04` |
| `1e-12` | `1.33e-09` | `1.355e-04` |
| `1e-14` | `1.45e-11` | `1.355e-04` |
| `1e-16` | `1.24e-13` | `1.355e-04` |

The subgradient difference falls by about `100×` for every two decades of
tolerance — proportional, with no floor down to `1.2e-13`. The regularisation
difference does not move at all, to four significant figures, across six
decades of tolerance.

This is the decisive result of the campaign. `sign(0) = 0` reaches the *same
root*, only less precisely when Newton is stopped early. `δ` reaches a
*different root*, and stopping Newton later does not bring it back.

Archived as `srix_semismooth_subgradient.json` and
`srix_semismooth_subgradient_epsilon_sweep.json`.

## H2 — `δ` does change the law

On the same records, `δ = 1e-5` differs from the historical run by `1.1e-4` to
`1.4e-4` overall, exceeding `1e-8` on the slip variables for 247, 217 and 161
of 380 records at the three fractions — far above the 10 % the hypothesis
required. Largest single family: `back_strain` at `4.9e-4`.

This matters for interpreting `δ`. In the archived failures the first non-zero
percentile of slip is `8.43e-6` and the fifth is `2.25e-5`, so `δ = 1e-5` is not
small against the physical slips it acts on. It is a constitutive modification
and the documentation must keep calling it one.

## P1 and P2 — the field differences come from the sub-steps

The M200 campaign reports `3.86e-5` on stress and `1.82e-4` on signed slip
between the historical baseline and the zero-subgradient run. Since the two
seek the same root, that cannot be the subgradient. The remaining candidate is
that the baseline composes 978 sub-stepped integrations and the new run
composes none.

Tested at the material point on the same 380 states, reproducing the bridge's
own sub-step sequence: **A** historical, sub-stepped as `_substep_span` would;
**B** zero-subgradient forced onto exactly the divisions A needed; **C**
zero-subgradient in one shot.

| quantity | A vs B — convention, same partition | B vs C — partition, same convention |
|---|---:|---:|
| stress | `4.56e-11` | `6.18e-03` |
| `elastic_strain` | `4.02e-11` | `1.02e-02` |
| `plastic_slip` | `1.00e-11` | `4.34e-03` |
| `equivalent_plastic_slip` | `1.00e-11` | `4.34e-03` |
| `back_strain` | `5.97e-11` | `1.53e-02` |

P1 required A vs B within `1e-8`: met with three decades to spare. P2 required
B vs C above `1e-6` on slip for at least 25 % of records: met for **380 of
380**.

**Eight orders of magnitude separate the two causes.** The subgradient choice
contributes `~6e-11`, itself tolerance-limited; discarding the sub-step
partition contributes `~1e-2` on these points. The campaign-level `1e-4` is the
L2 of that effect diluted over a mesh where most points never sub-stepped.

The historical convention needed only two divisions on 378 of the 380 points
and four on the remaining two — the failure is sharp, not deep.

Archived as `srix_substep_path_dependence.json`.

## What follows, and what does not

**Follows.** `SrixSlipZeroDerivative` is a choice of element in the Clarke
subdifferential `[-1, +1]` of `|Δγ|` at `Δγ = 0`, where the function has no
derivative. The historical `-1` is the left branch of `dg > 0 ? 1 : -1`, an
arbitrary side. It reaches the Jacobian only: in
`Fcc316LForestRubinSrix.mfront`, `dg_abs_derivative` is read at lines 303, 304
and 312, all writing `dfg_ddg`, and never by `feel`, `fg` or
`@UpdateAuxiliaryStateVariables`. Reading and measurement agree.

**Does not follow.** That the M200 fields under `sign(0) = 0` are *better*.
They are the same law integrated without sub-steps, which is a different
discretisation of the same evolution, not a more accurate one. Establishing
which is closer to the exact trajectory needs a refined-increment reference,
which has not been run.

**Does not follow.** Anything about 316L. The parameters remain
`316l_guilhem2013_nasri2018_meric_srix_rate_1e-3`, transposed from published
work, not identified on this material.

## Recommendation, and what changing the default would cost

Flipping `@Parameter real srixSlipZeroDerivative` from `-1.` to `0.` in both
`.mfront` sources is a one-line change that would remove 978 sub-stepped points
and two Newton iterations from the M200 case at no constitutive cost.

It is not done here. Every archived campaign was integrated with the historical
subgradient, and while the *root* is unchanged, the *sub-step partitions* are
not — so a replay under a flipped default would reproduce archived fields only
to `~1e-4`, not bit-for-bit, and every existing comparison would silently
acquire that floor. That is a decision about the reproducibility contract of
the archive, not a numerical question, and it belongs to the maintainer.

`tests/unit/core/test_srix_semismooth_subgradient.py` locks both facts the
recommendation rests on, whichever default is chosen.
