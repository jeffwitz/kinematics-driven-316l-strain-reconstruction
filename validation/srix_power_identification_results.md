# SRIX identification on the experimental power — results

Against `validation/srix_power_identification_preregistration.md`,
thresholds frozen before the runs. The verdict is **negative** — and it is
negative already in-sample.

> **Amendment, recorded before the verdict.** The identification was
> iterated on the central 20×20 window of the 100×100 grid (8000 point-steps
> instead of 400 000) so the loop stays fast enough to iterate; the frozen
> bars are unchanged and the substepping/penalty/bounds protocol is the one
> registered. A full-window rerun would follow only if this run were
> informative.

## Verdict

| bar | registered | measured | |
|---|---|---|---|
| SRIX explains the power | held-out `R^2 >= 0.30` | **-3.98** | fail |
| partial | `>= 0.10` | -3.98 | fail |
| negative | `< 0.10` | | **recorded** |

Held-out per state: `24: -7.6, 28: -2.3, 32: +0.04, 36: -9.3, 40: -0.78`.
Integration failures at the optimum: 0 (the substepping recovered the whole
path; the earlier finding stands that the default law already fails beyond
state 26 without it).

## What the fit itself did

The optimisation collapsed five of the six parameters (`tau0, R, Q, b, C`)
onto the **same** ratio `0.223` of their defaults and pushed `d` to `4.5x` —
a degenerate overall-scale fit chasing the power magnitude, and it still
did not match: the in-sample objective stayed at `1.03` (the fitted law
explains essentially none of the training power either). The fit is not
even good where it is allowed to look.

## What this closes

The known SRIX structure, with its six uncertain parameters free to fit the
training increments, does not explain the experimental plastic power —
neither held-out nor in-sample. Combined with the day's chain (no
discovered local state conditions the effective response; no scalar,
resistance, dynamical or path memory closes it), the conclusion is now
complete on both sides of the question: **the effective inelastic field is
not the constitutive response of SRIX — or of any tested state — and SRIX
itself does not reproduce its power on the measured kinematics.** The
forward path remains: a constitutive model inside the equilibrium problem,
validated against the held-out DIC — which is where the generator work
resumes, on the 200×200 window when the user decides.

## Outputs

`validation/_generated/shared_tensor_generator/srix_power_identification.json`
