# DIC-weighted FEMU observability evidence

**Mode:** reference  
**Domain:** evidence

This record describes a registered post-processing result.  It combines the
archived P43 displacement histories with the measured DIC transfer function
and repeated-frame uncertainty; it does not execute a new constitutive or
mechanical forward.

| Claim | Evidence ID | Primary artefact | Recorded configuration/result | Boundary |
|---|---|---|---|---|
| DIC-weighted observable modes can be excited in the archived histories | `E-DIC-OBSERVABILITY-001` | `validation/_generated/performance/experimental_oracle_p43_m20/dic_excitation_m20.json`; `.../dic_excitation_m100.json` | 40 states; 12 modes (M20), 20 modes (M100); seed 42; M20 expected noise norm 26.8701 and early max 0.4016 sigma; M100 expected norm 140.0071 and final mode-1 coefficient -118.0552 | Registered post-processing only; no new forward |
| The observation model uses measured transfer and noise, not a generic smoothing proxy | `E-DIC-OBSERVABILITY-001` | `validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv`; `validation/reference_data/dic_uncertainty_propagation_p0043_v1/centred_repeat_flow_pixels.npy` | Spectral transfer and repeated-frame whitener are applied by `scripts/project_dic_residuals_on_observable_modes.py` | The result is tied to the registered DIC chain |
| Observable residual modes are not a latent plastic-field reconstruction | `E-DIC-OBSERVABILITY-001` | `validation/dic_excitation_of_observable_plastic_modes.md` | Edge-dominated modes; elastic heterogeneity remains; corrected post-yield signal is a lower bound | No slip-system recovery or experimental parameter-identification claim |

The generated reports are:

- `validation/_generated/performance/experimental_oracle_p43_m20/dic_excitation_m20.json`
- `validation/_generated/performance/experimental_oracle_p43_m20/dic_excitation_m100.json`

The interpretive report is
`validation/dic_excitation_of_observable_plastic_modes.md`.  The evidence
registry contains the machine-readable claim and its provenance.  Together,
these artefacts support DIC-weighted observability for the registered M20/M100
histories; they do not authorise experimental 316L calibration or a
boundary-only production FEMU workflow.
