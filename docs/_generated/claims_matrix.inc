<!-- Generated from validation/documentation_evidence_registry.json. -->

| Question | Status | Claim boundary | Evidence |
|---|---|---|---|
| Micromorphic coupling measurably improves agreement with DIC | **demonstrated** | Amplitude, correlation and active area only, under symmetric observation on P43; localisation overlap degrades and no unique alpha is selected | `E-DIC-004` |
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
| An EBSD/Schmid structural correlation scale has been measured independently of FEM/DIC agreement | **verified** | Grain-mean maximum-Schmid map and declared 1.84 um grid; not a direct micromorphic or material length | `E-EBSD-001` |
| The declared DISFlow reproduction chain has a measured noise upper bound and spatial transfer | **verified for the reproduction implementation** | Candidate final-state repeat without acquisition log; OpenCV 4.14 reproduction, not bitwise historical DISFlow | `E-DIC-001` |
| The supplied legacy DISFlow source and the declared V4 profile have different measured transfer | **verified for OpenCV 4.14** | The legacy source leaves factory defaults implicit; this is provenance sensitivity, not certification of the unavailable historical binary | `E-DIC-003` |
| Symmetric image-level observation materially changes FEM/DIC metrics and candidate ranking | **verified** | Archived P43 displacement fields, legacy-source profile primary and declared V4 sensitivity; mechanics unchanged | `E-DIC-004` |
| Direct local brightness residual explains the remaining P43 FEM/DIC field error | **not supported** | One band-containing P43 core; direct unnormalised brightness residual; no propagated uncertainty claim | `E-DIC-005` |
| Hchi and ell statistically identified | **not demonstrated** | The earlier objective surface used raw FEM against observed DIC and must not be reused after the symmetric observation result | `E-ID-001`, `E-DIC-004` |
| Transferable material internal length identified | **not claimed** | Requires unchanged-parameter transfer to independent observations | `E-ID-001` |
| Measured temporal boundary history can be replayed end to end | **demonstrated** | Convergence only; says nothing about physical correctness of the intermediate states | `E-DIC-007` |
| Loading path changes reconstructed plasticity at a shared endpoint | **supported** | 15.8 % on core PEEQ under one constitutive model, invisible to the DIC observable; the DIC noise contribution is not subtracted | `E-DIC-008`, `E-DIC-010` |
| Measured path is closer to DIC than a proportional ramp | **not supported** | Indistinguishable at the registered margins; the observable is insensitive at this scale | `E-DIC-009` |
