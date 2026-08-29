# DIC-weighted tensor observability evidence

**Mode:** reference  
**Domain:** evidence

This record describes a registered **field-observability** post-processing
result.  It combines the archived P43 displacement histories with the measured
DIC transfer function and repeated-frame uncertainty; it does not execute a
new constitutive or mechanical forward and is not a parametric FEMU SVD.

| Claim | Evidence ID | Primary artefact | Recorded configuration/result | Boundary |
|---|---|---|---|---|
| Free tensorial inelastic fields have observable modes in the registered chain | `E-DIC-OBSERVABILITY-001` | `validation/_generated/performance/experimental_oracle_p43_m20/dic_excitation_m20.json`; `.../dic_excitation_m100.json` | 40 states; 12 modes (M20), 20 modes (M100); M20 expected noise norm 26.8701 and early max 0.4016 sigma; M100 expected norm 140.0071 and final mode-1 coefficient -118.0552 | Field/eigenstrain observability only; not parameter observability |
| Boundary masking does not recover plastic localisation | `E-DIC-OBSERVABILITY-002` | `validation/_generated/performance/experimental_oracle_p43_m20/mode_anatomy_m100/report.json`; `validation/_generated/performance/experimental_oracle_p43_m20/mode_anatomy_m100_interior/report.json` | 15-node mask: correlation +0.149 → -0.150; top-decile overlap 0.134 → 0.112 | Registered M100 geometry and DIC map |
| EBSD crystal elasticity does not explain the registered residual | `E-DIC-OBSERVABILITY-003` | `validation/_generated/performance/experimental_oracle_p43_m20/ebsd_elastic_reference_m100_polycrystal.json` | Rotated cubic FCC elasticity and shuffled control are nearly identical in residual norm; correction cosines +0.0015/+0.0057 | This tests one explanatory hypothesis, not the value of EBSD generally |
| Transfer choice materially changes modal amplitudes | `E-DIC-OBSERVABILITY-004` | `validation/_generated/performance/experimental_oracle_p43_m20/measurement_transfer_variants_m100.json` | Wrap-free transfer removes 57--71% of the tested residual; identity transfer remains sub-noise | No transfer-independent plastic-signal claim |

The generated reports are:

- `validation/_generated/performance/experimental_oracle_p43_m20/dic_excitation_m20.json`
- `validation/_generated/performance/experimental_oracle_p43_m20/dic_excitation_m100.json`

The interpretive report is
`validation/dic_excitation_of_observable_plastic_modes.md`.  The evidence
registry contains the machine-readable claim and its provenance.  Together,
these artefacts support DIC-weighted observability for the registered M20/M100
histories; they do not authorise experimental 316L calibration, parametric
FEMU observability, latent plastic-field recovery or a boundary-only production
FEMU workflow.
