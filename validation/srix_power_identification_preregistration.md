# SRIX identification on the experimental plastic power — preregistration

Registered before any run. Thresholds frozen. Negative results kept.

## Object

The phase-space discovery path is closed: no discovered local state
predicts the effective response across increments. The complementary
question — the one left open all along — is now asked directly: **does the
known SRIX structure, with its own internal variables, explain the
experimental plastic power?** The observable is the thermodynamically
central scalar the whole campaign has revolved around:

```text
D_n(x) = sigma_n(x) : Delta eps^inel_n(x)
```

per increment and per point of the reconstructed trajectories.

## Protocol

* **The law.** The compiled `Fcc316LForestRubinSrix` behaviour through the
  repo's own 3-D bridge, driven by the registered behaviour specification
  `316l_guilhem2013_nasri2018_meric_srix_rate_1e-3`, with the per-point
  EBSD orientation (`Q_global_to_material` from the Bunge maps, the repo's
  convention).
* **The path.** Strain-driven, on each point's reconstructed total-strain
  path (`eps^e + eps^inel_observable`, the kinematics the DIC actually
  constrains), states 21-40, 100×100 window, the zz completion from plane
  stress and plastic incompressibility.
* **The target.** `D_exp = sigma_pred : Delta eps^inel_observable`, the
  predictor-stress power of the observable increment — the same quantity
  the thermodynamic analyses used.
* **The parameters.** The six whose provenance claims no material
  knowledge: `tau0, R, Q, b, C, d`. Least squares over their logarithms
  (positivity enforced), central finite-difference gradients, L-BFGS-B,
  fitted on the 15 training states only.
* **The validation.** The five held-out states, never seen by the fit:
  per-state `R^2(D)` and the mean.

## Frozen bars

1. **SRIX explains the power.** Held-out mean `R^2(D) >= 0.30` — the bar
   no discovered state ever reached.
2. **Partial.** `0.10 <= R^2 < 0.30`: the known structure carries part of
   the power information.
3. **Negative kept.** `R^2 < 0.10`: the known SRIX structure does not
   explain the experimental power either — recorded as such, and the
   forward path (generator inside equilibrium, validated on the held-out
   DIC) is confirmed as the only remaining route.
4. The fitted parameters are reported as ratios to the default set; a
   parameter pushed to a bound is flagged.

## Registered caveats

The power is a scalar summary: this test asks whether SRIX explains the
*amount* of dissipation, not the directional structure (which the day's
analyses showed the effective field does not carry). A success here is
therefore necessary but not sufficient for the full law — and a failure
is decisive.

## Outputs

`validation/_generated/shared_tensor_generator/srix_power_identification.json`
and this file's results companion.
