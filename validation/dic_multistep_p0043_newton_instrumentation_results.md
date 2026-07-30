# P43 measured-history Newton instrumentation — results

Date: 2026-07-30
Preregistration: `dic_multistep_p0043_newton_instrumentation_preregistration.md`.
Trace: `reference_data/dic_multistep_newton_trace_p0043_v1/newton_trace.csv`,
28 records, produced by a run whose campaign and prepared-case manifests carry
the same SHA-256 as the archived failure.

## Short answer

The failure is a **software defect, not a numerical or physical one**.

The elastic predictor of the measured-history branch is solved against a CSR
buffer that the elastoplastic tangent assembly has already overwritten. From
the first increment in which plasticity makes the tangent differ materially
from the elastic operator, the predictor is computed with the wrong matrix. The
corruption compounds at every cutback, which is why smaller steps make the
failure worse rather than better.

The three registered hypotheses were all aimed at the linear algebra of a
correctly posed problem. None of them is the cause.

## The registered discriminator

| Test | Measured | Fires? |
|---|---|---|
| tangent diagonal non-positive or near-zero | count is `0` at every iteration | no |
| `max\|D_ep\| / max\|C_elastic\| > 1 + 1e-6` | exactly `1.000000` at every iteration | no |
| both clean and `\|\|du\|\| / \|\|du_B\|\| < 1e3` | `2.57e-03` at increment 4 iteration 1 | **yes** |

Read literally the discriminator selects H3, genuine globalisation. That
reading is wrong, and the trace shows why: the registered H1 test was an
inadequate proxy. A matrix can have a strictly positive diagonal and still be
arbitrarily ill-posed, and the operator actually used by the predictor is not
the one whose diagonal was inspected. H2 is cleanly and correctly refuted: the
consistent tangent never exceeds the elastic operator.

## What the trace shows

Increment 4, the transition from state 3 to state 4, at `dt = 2.50e-02`:

| Iteration | residual | `\|\|du\|\|` | tangent diagonal min | max total strain |
|---:|---:|---:|---:|---:|
| 1 | `1.661` | `2.99e-04` | `1.974e+04` | `1.855e-02` |
| 2 | `2.192` | `1.02e-03` | `6.341e+03` | `6.959e-02` |
| 3 | `2.902` | `4.50e-03` | `2.183e+03` | `3.069e-01` |
| 4 | `3.905` | `1.96e-02` | `8.078e+02` | `1.345` |
| 5 | `5.289` | `8.08e-02` | `3.167e+02` | `5.598` |
| 6 | `7.070` | `3.13e-01` | `1.230e+02` | `2.200e+01` |
| 7 | rejected | — | — | `8.226e+01` |

For comparison, increment 3 converged in five iterations with a tangent
diagonal minimum of `3.3e+05` to `4.0e+05`.

Two observations decide the diagnosis.

**The first iteration of increment 4 is already wrong.** Its tangent diagonal
minimum is `1.974e+04`, seventeen times softer than increment 3, *before* any
divergence. Its trial strain is `1.855e-02` where a linear extrapolation from
increment 3 predicts about `8.7e-04`, a factor of twenty-one.

**Increments 5 to 11 fail at iteration 1 with a trial strain exactly
proportional to the step size:**

| Increment | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dt` | `1.25e-02` | `6.25e-03` | `3.13e-03` | `1.56e-03` | `7.81e-04` | `3.91e-04` | `1.95e-04` |
| max strain | `4423` | `2211` | `1106` | `553` | `276` | `138` | `69.1` |

Exact halving across seven increments is the signature of a **frozen linear
operator**. These increments never reach a tangent assembly, so whatever matrix
the predictor uses stays fixed while only `du_B` shrinks. A correct elastic
predictor would give a trial strain near `1e-03`, not `4423`.

## Root cause

`FixedCSRAssembler.assemble` is documented as returning the same CSR object
with `matrix.data` updated in place:

```python
np.copyto(self.matrix.data, sums[: self.matrix.nnz])
return self.matrix
```

Verified directly: two successive `assemble` calls return the same object and
the first result is mutated by the second.

In `run_fem` both the elastic stiffness and every elastoplastic tangent come
from the same assembler:

```python
KII_el = fixed_free_assembler.assemble(Ke)       # elastic
K_tang = fixed_free_assembler.assemble(Ke_ep)    # same object, same buffer
```

so `K_tang is KII_el`. After the first elastoplastic assembly, `KII_el` holds
the elastoplastic tangent, and `solve_el` no longer solves an elastic problem.

## Why only the measured history fails

`solve_el` is used in exactly two places:

- before the increment loop, for the proportional path:
  `elastic_predictor_direction = solve_el(-KIB_el @ u_bc[dof_B])`, evaluated
  once while the buffer still holds the elastic operator, then merely scaled by
  `dt`;
- **inside** the increment loop, for the measured-history path:
  `elastic_free_increment = solve_el(-KIB_el @ du_B)`.

Only the history branch re-solves each increment, and only it meets the
corrupted buffer. The proportional baseline converges because its predictor was
computed before any plastic assembly.

The apparent contradiction recorded in `Claude.md` — the same partition
converging on a proportional ramp and failing on the measured path — therefore
has a software explanation and does not require a physical one. The earlier
reasoning about the two paths crossing plastic activation differently remains
true as a statement about the loading, but it is not the cause of the failure.

Every archived observation is accounted for:

| Observation | Explanation |
|---|---|
| fails at the first substantial plastic transition | first increment where the plastic tangent departs materially from elastic |
| proportional baseline converges | its predictor is computed before corruption |
| smaller steps fail earlier | each failed increment leaves a more diverged tangent in the buffer |
| increments 5 to 11 scale exactly with `dt` | frozen corrupted operator, no new assembly |
| failing elements sit on the boundary | the predictor error is driven by `-KIB_el @ du_B` |
| rejected strains `3.6e5` times any measured strain | the predictor is not a mechanical field |

## Scope of the defect

No archived scientific result is affected. Every archived campaign uses the
proportional path, whose predictor is computed once before the buffer is
overwritten. Only measured-history runs are affected, and all of them already
failed. Correcting the defect therefore cannot invalidate any published number.

## Consequences

1. The multistep blockage is reclassified from a nonlinear-convergence
   limitation to a defect in the elastic predictor of the history branch.
2. The line-search campaign that ran `2 h 47` and the frame-removal attempts
   were all treating a symptom. So was the boundary-noise hypothesis refuted in
   `dic_boundary_loading_subspace_p0043_results.md`.
3. The fix and the rerun are registered separately in
   `dic_multistep_p0043_predictor_fix_preregistration.md`. Until that rerun
   completes, no claim is made that the measured history converges.
4. The registered discriminator of this campaign is recorded as inadequate: a
   positive diagonal does not establish a well-posed operator, and the operator
   inspected was not the one the predictor used.

## Reproduction

```bash
fem-inhouse run-dic-multistep-mechanics \
  --prepared-case data/processed/case_study \
  --source-campaign results/constitutive-local-p0043-pad150 \
  --history validation/reference_data/dic_multistep_history_p0043_repaired_v1 \
  --partition-id 43 --mode measured --record-newton-trace \
  --output validation/reference_data/dic_multistep_newton_trace_p0043_v1
```
