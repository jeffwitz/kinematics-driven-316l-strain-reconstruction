# Preregistration — sequential one-correction SRIX diagnostic

## Question

Does one tangent displacement correction per increment, followed by a
constitutive re-evaluation and commit, restore the local FEMU sensitivity
geometry that the fixed-history REGM misses?

## Frozen method

Use the exact M8 SRIX twin and the same four log-parameter central finite
differences (`h=3e-3`) as the archived information-geometry study. At each
increment, extrapolate the predictor from the previously accepted corrected
state plus the measured twin increment. Evaluate SRIX with a consistent
algorithmic tangent, assemble the weak residual, solve one correction, then
revert the trial, re-evaluate at the corrected displacement, and commit. Never
iterate global Newton to convergence. Score the affine-preserving observation
of the one-step correction at the registered states.

Compare the normalized SVD, condition number and principal angles with the
archived `REGM-K0`, `REGM-Kalg` and observed FEMU Jacobians. Do not use P43 or
launch new full forward perturbation solves.

## Decision rule

This is a diagnostic, not an identification run. It is promising only if the
rank-two principal angle to FEMU decreases substantially and the weak singular
directions are materially larger than in fixed-history REGM. Otherwise stop
REGM development and retain the NO-GO before P43.
