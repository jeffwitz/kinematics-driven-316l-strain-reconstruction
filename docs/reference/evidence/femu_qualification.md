# FEMU qualification evidence

**Mode:** reference  
**Domain:** identification

The registered evidence separates a working synthetic identification chain from
the still-missing experimental production workflow.

| Claim | Evidence ID | Configuration | Recorded result | Boundary |
|---|---|---|---|---|
| direct synthetic FEMU smoke | `E-SRIX-P43-SYNTH-001` | P43 M20, real EBSD, identity observation, no noise, four log-parameters | RMS `8.8502e-8 → 1.2693e-13`; six-evaluation smoke; `Q/b` correlation `0.999999997` | positive synthetic smoke; four-parameter recovery not claimed |
| distant-start synthetic identification | `E-SRIX-P43-SYNTH-002B` | P43 M20, four deterministic starts, no noise | demonstrated; the weak `Q/b` valley remains | no experimental claim |
| M20-to-M100 synthetic scale-up | `E-SRIX-P43-SYNTH-003` | P43 M100, M20 initialisation, identity observation, 32 path steps, `h=0.0015` | three evaluations; RMS `3.38e-16 → 3.25e-18`; parameters recover synthetic truth to numerical precision | one registered scale-up; `Q/b` remains correlated |
| experimental P43 identification | `E-SRIX-P43-EXP-001-M20`, `E-SRIX-P43-EXP-RAW-001` | experimental P43 M20, registered observation and SVD parameterisations | NO-GO / constrained valley; no identification authorised | experimental observability remains limited |

The primary machine-readable records are in
`validation/documentation_evidence_registry.json`. A synthetic recovery
validates the sensitivity and optimisation machinery; it does not establish a
material calibration. The registered
`scripts/srix_femu_smoke.py` driver is instead a full-field-Dirichlet negative
control and is not a production boundary-only How-to.
