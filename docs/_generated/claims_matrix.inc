<!-- Generated from validation/documentation_evidence_registry.json. -->

| Question | Status | Claim boundary | Evidence |
|---|---|---|---|
| Finite-element discretization verified | **verified** | Structured small-strain plane-stress Q4 scope | `E-FE-001` |
| Historical Abaqus-oriented table definition implemented in InHouse/Table | **verified** | Constitutive table definition only; this is not an Abaqus mesh, boundary-condition or ODB parity claim | `E-TABLE-001` |
| Complete Abaqus model reproduced | **not demonstrated** | Original input model, ODB and extraction procedure are unavailable | `E-TABLE-001` |
| InHouse/Table and analytical MFront response coherent | **verified** | Declared material-point and finite-element tolerances below the table cap | `E-LOCAL-001`, `E-LOCAL-002` |
| Micromorphic interaction redistributes plasticity | **verified** | Converged band-containing reconstruction campaigns at fixed length | `E-NL-002` |
| Spatial length affects morphology independently of coupling strength | **supported** | Reduced-fidelity constant-Achi and fixed-alpha experiments; uncertainty envelopes and F2 transfer remain absent | `E-ID-001` |
| Hchi and ell statistically identified | **not demonstrated** | Short-length high-coupling region remains numerically censored | `E-ID-001` |
| Transferable material internal length identified | **not claimed** | Requires unchanged-parameter transfer to independent observations | `E-ID-001` |
