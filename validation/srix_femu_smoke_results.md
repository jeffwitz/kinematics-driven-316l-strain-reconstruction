# SRIX FEMU-U smoke — results

Against `validation/srix_femu_smoke_preregistration.md`. The loop is
operational; the observable is degenerate; the verdict is a **wiring
finding**, not a law verdict.

## Verdict

| quantity | measured |
|---|---|
| fitted parameters | `tau0 = 40.0`, `Q = 10.0` — the fit did not move |
| misfit, default SRIX | `1.088e-9` |
| misfit, fitted SRIX | `1.088e-9` |
| misfit, J2 baseline | `3.63e-9` |
| reading | negative — **degenerate observable** |

## What happened, and what it means

On the historical band window — with real plasticity content — every law
(SRIX default, fitted, J2) reproduces the measured displacement to the
solve tolerance (`~1e-9`). The misfit carries no constitutive sensitivity,
so the L-BFGS gradient was flat and the parameters never moved. The cause
is the boundary-condition contract of the partition workflow: the prepared
case's measured displacement is imposed as the **full nodal field** (the
solver's `boundary_displacement_history_mm` takes the complete nodal shape
and eliminates it), so the interior has no freedom and the solved field is
the measured field for any law.

This is consistent with the day's own architecture, where the measured
kinematics were always imposed on the **boundary** of the reconstruction
window with the interior free — the FEMU-U loop as wired here skipped that
step. The comparison instrument that does have constitutive sensitivity
(and has been used all day) is the boundary-driven reconstruction, not the
full-field-Dirichlet partition run.

## What works now

The full plumbing of the forward loop is operational: the condensed SRIX
backend with canonical parameter overrides (`tau0_mpa`, `Q_mpa`), the
prepared historical-window case, manifest-safe evaluation directories, the
FD-gradient L-BFGS loop, the stitched U field, and the J2/SRIX references.

## The fix, next

Impose the measured displacement on the **boundary nodes only** (interior
free) — either by preparing a case whose displacement files carry the
measured field on the edge with the interior left to the solve (the
workflow's non-finite handling exists), or by a boundary-only extraction
in the workflow — then rerun the same loop. Only then does the FEMU-U
reading become a statement about SRIX.
