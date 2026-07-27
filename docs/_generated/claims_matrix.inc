<!-- Generated from validation/documentation_evidence_registry.json. -->

| Question | Status | Claim boundary | Evidence |
|---|---|---|---|
| Finite-element baseline passes its numerical regression contract | **verified** | Structured small-strain plane-stress Q4 integration scope | `E-LOCAL-002` |
| Generalized section equilibrium has been quantified on the archived band-containing cases | **reported baseline** | Interior P43 partition; lateral shear flux included; no experimental force validation | `E-EQ-001` |
| The pixel-wise material maps add spatial localisation information beyond boundary kinematics | **supported** | One band-containing ROI; translated-map and homogeneous controls; no transfer claim | `E-MAP-001` |
| Available Abaqus-oriented table definition reimplemented | **verified at definition level** | The versioned 1000-row generator is checked; Abaqus FE and ODB parity are excluded | `E-TABLE-001` |
| Abaqus/Table and InHouse/Table finite-element results agree | **not demonstrated** | Original input model, ODB and extraction procedure are unavailable | `E-TABLE-001` |
| InHouse/Table and InHouse/MFront are coherent | **verified** | Declared material-point and finite-element comparison thresholds | `E-LOCAL-001`, `E-LOCAL-002` |
| A 3D MFront law can be condensed without changing the 2D solver | **verified** | Native plane stress versus 3D-condensed J2 on the reduced DIC case | `E-3D-001` |
| Output-only spatial broadening supports a missing-width hypothesis | **supported** | Diagnostic only; mechanics is unchanged and no material length is identified | `E-DIAG-001` |
| Micromorphic interaction redistributes plasticity | **verified** | Converged full-resolution band-containing reconstruction campaigns | `E-NL-002` |
| Spatial length affects morphology independently of coupling strength | **supported** | Validated reduced-fidelity constant-Achi and fixed-alpha experiments | `E-ID-001` |
| Hchi and ell statistically identified | **not demonstrated** | The short-length high-coupling region remains numerically censored | `E-ID-001` |
| Transferable material internal length identified | **not claimed** | Requires unchanged-parameter transfer to independent observations | `E-ID-001` |
