# SRIX FEMU-U smoke — preregistration

Registered before any run. Thresholds frozen. Negative results kept.

## Object

The forward path in its minimal form: the SRIX law inside the equilibrium
problem, parameters adjusted by least squares on the measured displacement
(FEMU-U), with a **small** parameter freedom. The question is only whether
freeing two parameters and fitting displacements helps at all.

## Protocol

* **Window.** A 20×20-element prepared case from the raw case data (the
  prepare-case crop starts at the raw origin — the smoke window is a corner
  of the ROI, not the historical band zone; this is a wiring-and-signal
  smoke, not the scientific run). Identity crystal orientation — the
  simplest admissible SRIX run; the EBSD orientation is a later stage.
* **Law.** `mfront-srix-generic-plane-stress` with the default parameter
  set, free parameters `tau0` and `Q` (log space, central finite
  differences, L-BFGS-B, bounds ±1.5 around the defaults).
> **Amendment, recorded before the run.** The prepared case stores the
> measured displacement as a single final field, so the observable is the
> **final-state** displacement misfit `J(theta) = |u_theta - u_meas|^2`
> and the held-out-increment protocol of the day does not apply to this
> smoke; the fit and the bars read the final state only. The reference is
> the python J2 run of the same case (the day's baseline law), not an
> elastic solution.

* **Observable.** `J(theta) = |u_theta - u_meas|^2` on the final measured
  state, `u_theta` from the stitched U field.
* **Reference.** The python J2 backend on the same case, same BCs.

## Frozen bars

1. **FEMU-U helps.** The fitted SRIX's final-state misfit is `<= 0.9 x`
   the J2 baseline's.
2. **FEMU-U beats the default SRIX** by at least `10 %` of its misfit.
3. Anything below is the recorded negative: freeing `(tau0, Q)` and
   fitting the displacement does not help this law on this window — the
   parameter freedom would have to be different before any scientific run.

## Outputs

`validation/_generated/shared_tensor_generator/srix_femu_smoke.json` and
this file's results companion.
