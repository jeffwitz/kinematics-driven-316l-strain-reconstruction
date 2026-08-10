# Preregistration — is `sign(0) = 0` a change of law, or a change of subgradient?

Frozen before execution. Thresholds and falsifiers below are not to be moved
after seeing results.

## Question

`SrixSlipZeroDerivative` selects the value assigned to `d|Δγ|/dΔγ` at exactly
`Δγ = 0`. The historical law uses `-1`, the left-hand branch of
`dg > 0 ? 1 : -1`. The control experiment uses `0`, the symmetric element of
the Clarke subdifferential `[-1, +1]`.

On the 380 archived isolated failures of P43 M200, the historical value fails
380/380 and `0` rescues 380/380. On the full M200 campaign the historical run
needs 978 sub-stepped points and 58 Newton iterations; `sign(0) = 0` needs 0
and 56.

Two readings of that are possible and they have opposite consequences:

- **A — it changes the law.** Then `sign(0) = 0` is a constitutive variant on
  the same footing as the compact regularisation `δ`, and it must be qualified
  as one, with its own provenance and an explicit warning against comparing its
  results to archived campaigns.
- **B — it changes only the Jacobian.** Then the residual, and therefore the
  root being sought, is untouched. `sign(0) = 0` would be a *correction of the
  semismooth Newton subgradient at the cusp*, and could become the default with
  a compatibility switch, because it does not alter what the law computes.

Reading the source supports B: in `Fcc316LForestRubinSrix.mfront`,
`dg_abs_derivative` is consumed only at lines 303, 304 and 312, all of which
write `dfg_ddg` or feed `dda_ddg`, which itself writes `dfg_ddg(i, i)`. It
never reaches `feel`, `fg`, or `@UpdateAuxiliaryStateVariables`. The quantity
that *does* enter the residual, `dg_abs_regularized`, is exactly `abs(dg)` when
`δ = 0`, independently of `SrixSlipZeroDerivative`.

That is an argument from reading, not a measurement. This experiment measures
it, and measures the contrast against `δ`, which by the same reading *does*
change the residual through `p_`, `da` and `x`.

## Protocol

Source of states: the 380 archived isolated failures,
`srix_p43_m200_failure_diagnostics.failure_diagnostics.npz`, replayed one
material point at a time through the existing harness, `s0` transplanted
verbatim.

These are states on which the historical convention *fails* at the full
increment, so the comparison cannot be made there — a failed integration has no
converged state to compare. Each record is therefore also replayed at reduced
increment fractions

```text
lambda in {0.125, 0.25, 0.5, 1.0}
target(lambda) = s0_in_plane + lambda * (target - s0_in_plane)
```

which is the same device sub-stepping uses to get these points to converge. For
small enough `lambda` the historical convention converges, and only records
where **both** conventions converge enter the comparison.

Three variants, identical in every other respect:

| label | `SrixSlipSmoothingDelta` | `SrixSlipZeroDerivative` |
|---|---:|---:|
| `historical` | `0` | `-1` |
| `zero_subgradient` | `0` | `0` |
| `compact_delta` | `1e-5` | `-1` |

Compared quantities, at the converged state: the six thermodynamic forces and
all 46 internal state variables, as a relative L2 difference against the
historical run of the same record and the same `lambda`.

## Hypotheses and thresholds

**H1 — `sign(0) = 0` does not move the root.** On every record where both the
historical and the zero-subgradient variants converge, the relative difference
is at most `1e-11` on stress and on every internal state variable.

The local Newton criterion is `@Epsilon 1.e-14`, so two runs sharing a residual
but differing in Jacobian should agree far inside this bound. `1e-11` leaves
three decades of margin for iterate-path effects and is still four decades
tighter than the field differences the M200 comparison reports (`1e-4`).

*Falsifier:* any record exceeding `1e-11`. That would mean the residual is not
in fact independent of the subgradient choice, my reading of the source is
wrong, and reading A holds.

**H2 — `δ = 1e-5` does move the root.** On the same records, the compact
regularisation differs from the historical run by more than `1e-8` relative on
the slip variables, for at least 10 % of records.

*Falsifier:* `δ` also agrees within `1e-11` everywhere. That would mean this
state population cannot distinguish a constitutive modification from a
subgradient one, and the experiment is uninformative rather than supportive —
it would not vindicate `δ`, it would invalidate the test.

**H3 — the rescue is not an artefact of the reduced increment.** At
`lambda = 1.0` the historical convention still fails on 380/380 while the
zero-subgradient variant converges on 380/380, reproducing the archived replay.

*Falsifier:* a materially different failure count, which would mean the
transplant in this script does not reproduce the archived one and no conclusion
may be drawn from it.

## What this experiment does not settle

It is a material-point experiment. It cannot explain the `~1e-4` field
differences of the M200 campaign, which are a property of the global
trajectory: the historical baseline reaches its answer through 978 sub-stepped
points and the zero-subgradient run through none, so the two integrate the same
constitutive evolution along different discrete paths. Separating that requires
forcing the zero-subgradient run onto the historical sub-step partition, which
is a separate experiment and is not preregistered here.

Nothing here identifies 316L. The parameters remain
`316l_guilhem2013_nasri2018_meric_srix_rate_1e-3`, transposed from published
work.

---

# Preregistration, part 2 — do the field differences come from sub-stepping?

Frozen before execution, after part 1 was measured and before any part-2 run.

## Question

Part 1 established that the two conventions seek the same root: the difference
between their converged states falls proportionally with the local Newton
tolerance and reaches `1.24e-13` at `epsilon = 1e-16`, while the compact
regularisation stays at `1.355e-04` regardless of the tolerance.

The M200 campaign nevertheless reports `3.86e-5` on stress and `1.82e-4` on
signed slip between the historical baseline and the zero-subgradient run. If
the root is the same, those differences cannot come from the subgradient
choice. The remaining candidate is the integration path: the baseline reaches
its answer through 978 sub-stepped points and the zero-subgradient run through
none, so the two compose the same constitutive evolution over different
discrete partitions of the increment.

This is tested at the material point, on the 380 archived states, by
reproducing the bridge's own sub-step sequence explicitly. No global campaign
and no change to the solver is involved.

## Protocol

For each archived record, at the full increment:

- **A** — historical convention, sub-stepped. The number of divisions is found
  as `_substep_span` finds it: `2, 4, 8, ...`, doubling until the whole
  sequence converges, capped at 1024. The total in-plane strain is interpolated
  between the committed state and the target, and the state is advanced between
  sub-steps.
- **B** — zero-subgradient convention, forced onto **exactly the divisions A
  used**, same interpolation, same advance.
- **C** — zero-subgradient convention, one shot, no sub-stepping.

Compared at the final state: the six thermodynamic forces and the declared
internal-state families, as relative L2 differences.

## Hypotheses and thresholds

**P1 — on the same partition, the two conventions agree.** `A` and `B` differ
by at most `1e-8` relative on stress and on every internal-state family.

The bound is looser than part 1's per-step figure because a sequence of `n`
sub-steps composes `n` tolerance-limited solutions: at `epsilon = 1e-14` one
step contributes about `1.5e-11`, so a 1024-step sequence admits `1.5e-8` in
the worst case of no cancellation. `1e-8` remains four decades below the effect
being explained.

*Falsifier:* `A` and `B` differ by more than `1e-8`. The subgradient choice
would then matter beyond the tolerance once composed, and part 1's conclusion
could not be carried to the campaign scale.

**P2 — removing the sub-steps is what moves the answer.** `B` and `C` differ by
more than `1e-6` on signed slip for at least 25 % of the records that
sub-stepped in `A`.

*Falsifier:* `B` and `C` agree within `1e-8`. The sub-step partition would then
not be the origin of the campaign differences, and the `1e-4` seen at M200
would remain unexplained — leaving open that it comes from somewhere else
entirely, which would have to be found before `sign(0) = 0` could be proposed
as a default.

## Verdict rule, fixed in advance

`sign(0) = 0` may be recommended as the default convention only if P1 and P2
both hold, in addition to part 1's H2 and H3. If P1 fails, it stays an
experimental variant. Neither outcome licenses a claim about 316L: the
parameters remain transposed, not identified.
